"""Process route tables.

Join straight line sigments and attributes to reduce data size.

Created on Wed Oct 27 09:50:41 2021

@author: twillia2
"""
import os
import warnings

from glob import glob

import geopandas as gpd
import numpy as np
import pandas as pd
import pathos.multiprocessing as mp
import shapely.ops as ops

from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.geometry.collection import GeometryCollection
from shapely.ops import cascaded_union, nearest_points
from tqdm import tqdm

tqdm.pandas()
pd.set_option("display.max_columns", 50)
pd.options.mode.chained_assignment = None
warnings.simplefilter("ignore", ShapelyDeprecationWarning)


HOME = os.path.abspath("..")
DSTDIR = os.path.join(HOME, "data/shapefiles/routes/merged")
KEEPERS = ["FromNodeID", "ToNodeID", "StreetNameBase", "UATYP10",
           "NAME10", "DirOnSign", "LANE_CATEGORY", "geometry"]
COLUMNS = ["from_id", "to_id", "street", "urban_cat", "urban_name",
           "direction", "lane_count", "geometry"]


def cplot(*shapes, title=None, zoom=False, **kwargs):
    """Plot a collection of shapes."""
    df = gpd.GeoDataFrame({"geometry": shapes,
                           "value": [i for i in range(len(shapes))]})
    ax = df.plot("value", categorical=True, legend=True, **kwargs)

    if zoom:
        xmin, ymin, xmax, ymax = shapes[0].bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    if title:
        ax.set_title(title)


class Buffer:
    """Methods for buffering route segments."""

    def __init__(self, home=HOME, distance=65):
        """Initialize Characterize object."""
        self.home = os.path.expanduser(os.path.abspath(home))
        self.distance = distance

    def build_path(self, direction):
        """Build target file path for buffered dataset."""
        subdir = os.path.join(self.home, "data/shapefiles/routes/buffered")
        tpath = f"{direction}_interstates_buffered.gpkg"
        os.makedirs(subdir, exist_ok=True)
        dst = os.path.join(subdir, tpath)
        return dst

    def buffer_dataset(self, df):
        """Buffer an entire data frame."""
        # Apply Buffering
        cdf = df.copy()

        # We are missing some opposing geometries
        cdf = cdf[cdf["op_geometry"].length > 0]

        # Build initial buffers
        cdf = self._par_apply(cdf, self.buffer_segment)

        # Unpack the MultiPolygons
        entries = []
        for i, entry in tqdm(cdf.iterrows(), total=cdf.shape[0]):
            geom = entry["geometry"]
            if geom:
                if isinstance(geom, MultiPolygon):
                    for poly in geom:
                        if isinstance(poly, MultiPolygon):
                            raise ValueError("MultiPolygon found.")
                        sub_entry = entry.copy()
                        sub_entry["geometry"] = poly
                        entries.append(sub_entry)
                else:
                    entries.append(entry)

        cdf2 = gpd.GeoDataFrame(entries, crs=cdf.crs).reset_index(drop=True)
        cdf2["sid"] = cdf2.index

        # Create buffer joints
        cdf3 = self.buffer_joint(cdf2)

        # Temp
        del cdf3["op_geometry"]
        # cdf.to_file("/scratch/twillia2/test.gpkg", "GPKG")

        return cdf3

    def buffer_segment(self, entry, plot=False):
        """Shift and buffer segment in the appropriate direction.

        Production Notes:
            - Infer which side the one-sided buffer lands on
            - Use center line to create a full buffer (with endcaps) for the
            shifted buffer
        """
        # Extract geomtery, number of lanes, and direction        
        segment = entry["geometry"]
        op = entry["op_geometry"]  # Opposite interstate
        lanes = entry["lane_count"]
        width = lanes * 3.6576  # number of lanes times 12 feet
        buff_dist = self.distance / 2
        offset = (width / 2) + buff_dist

        # It's possible that the segment is a single line string
        if isinstance(segment, LineString):
            segment = [segment]

        # Apply offset and buffer to each geometry in segment
        tbuffers = {}
        sbuffers = []
        for i, line in enumerate(segment):
            sbuffer = self.buffer_line(line, op, offset, buff_dist)
            tbuffers[i] = sbuffer
            sbuffers.append(sbuffer)

        # Repackage and merge geometries into final buffer
        mrow = MultiPolygon(sbuffers)
        row = cascaded_union(mrow)

        return row

    def buffer_line(self, line, op, offset, buff_dist):
        """Determine the correct side to offset a road."""
        # Find the nearest point on the opposite line
        mid_point = line.interpolate(0.5, normalized=True)
        lp, opp = nearest_points(mid_point, op)
        bridge = LineString((lp, opp))

        # Attempt a positive offset and buffer 
        test_offset = line.parallel_offset(1)

        # If it intersects the offset was in the wrong direction
        if test_offset.intersects(bridge):
            offset *= -1

        # Then build the offset and buffer
        oline = line.parallel_offset(offset)
        buffer = oline.buffer(buff_dist)
    
        return buffer

    def buffer_joint(self, cdf):
        """Create a joint between connecting buffer segments."""
        ndf = cdf.copy()
        entries = []
        for i, entry in tqdm(ndf.iterrows(), total=ndf.shape[0]):
            g1 = entry["geometry"]
            g2s = ndf["geometry"][ndf["sid"] != entry["sid"]].values
            g2s = [g for g in g2s if g.intersects(g1)]
            if g2s:
                for g2 in g2s:
                    if g1.intersects(g2):
                        try:
                            g1 = self.cut(g1, g2)
                        except:
                            print("Cut Problem Found.")
                            pass
            entry["geometry"] = g1
            entries.append(entry)
        df = gpd.GeoDataFrame(entries)
        df.crs = ndf.crs
        return df

    def cut(self, g1, g2):
        """Second gen cut."""
        try:
            p1, p2 = g1.exterior.intersection(g2.exterior)
            cutline = LineString((p1, p2))
        except ValueError:
            ps = g1.exterior.intersection(g2.exterior)
            cutline = LineString(ps)

        # And cut with the cutline
        dg = g1.difference(cutline.buffer(1e-5))

        # Which side? Using largest area for now
        try:
            max_area = np.max([g.area for g in dg])
            final = [g for g in dg if g.area == max_area][0]
            return final
        except:
            print("Problem Found")
            return g1


class Merger(Buffer):
    """Methods for merging route segments."""

    def __init__(self, home=".", dstdir=".", distance=65):
        """Initialize Merger object."""
        super().__init__(home, distance)
        self.dstdir = os.path.abspath(os.path.expanduser(dstdir))
        
    def __repr__(self):
        """Return Characterize object representation string."""
        attrs = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
        msg = f"<Merger object: {attrs}>"
        return msg

    def attributes(self):
        """Merge attributes."""

    def connect_lines(self, df):
        """Connect lines that touch and share attribute values."""
        crs = df.crs
        df = df.sort_values("from_id").reset_index(drop=True)
        df["sid"] = df.index
        rows = []
        skippers = []
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            sid = row["sid"]
            if sid not in skippers:
                to = row["to_id"]
                if to in df["from_id"].values:
                    row2 = df[df["from_id"] == to].iloc[0]
                    q1 = row["street"] == row2["street"]
                    q2 = row["direction"] == row2["direction"]
                    q3 = row["urban_name"] == row2["urban_name"]
                    q4 = row["lane_count"] == row2["lane_count"]
                    if q1 and q2 and q3 and q4:
                        g = ops.unary_union([row["geometry"], row2["geometry"]])
                        row["geometry"] = g
                        skippers.append(row2["sid"])
                rows.append(row)
        df = gpd.GeoDataFrame(rows)
        df.crs = crs
        return df

    @property
    def files(self):
        """List route files."""
        files = glob(os.path.join(self.home, "data/shapefiles/routes/*ua.gpkg"))
        ew = [f for f in files if "ew" == os.path.basename(f)[:2]][0]
        ns = [f for f in files if "ns" == os.path.basename(f)[:2]][0]
        files = {"ns": ns, "ew": ew}
        return files

    def get_opposite(self, df):
        """Find, clip, and attach the opposite interstate segments."""
        opposites = {"N": "S", "S": "N", "W": "E", "E": "W"}
        ddfs = []
        for direction in df["direction"].unique():
            opposite = opposites[direction]
            ddf = df[df["direction"] == direction]
            vector2s = df["geometry"][df["direction"] == opposite].values
            vector2 = GeometryCollection(vector2s)
            ddf["op_geometry"] = self._par_apply(
                ddf["geometry"],
                self._get_opposite,
                vector2=vector2
            )
            ddfs.append(ddf)
        df = pd.concat(ddfs)
        return df

    def merge(self, file):
        """Merge continuous line segments."""
        # Read in and subset table
        df = gpd.read_file(file)
        tdf = df[KEEPERS]
        tdf.columns = COLUMNS

        # Remove major metropolitan urban centers and directionless segments
        tdf = tdf[tdf["direction"] != ""]
        # tdf = tdf[tdf["urban_cat"] != "U"]
        tdf.loc[pd.isnull(tdf["urban_name"]), "urban_name"] = "none"
        tdf = tdf[~tdf["geometry"].isnull()]

        # Fix inconsistencies
        tdf.loc[tdf["street"] == "I 40", "street"] = "I-40"
        tdf.loc[tdf["street"] == "I 35", "street"] = "I-35"
        tdf.loc[tdf["street"] == "Ih 35", "street"] = "I-35"
        tdf.loc[tdf["street"] == "Interstate 35", "street"] = "I-35"
        tdf.loc[tdf["street"] == "I 10 Frontage", "street"] = "I-10 Frontage"
        tdf.loc[tdf["street"] == "Interstate 10", "street"] = "I-10"
        tdf.loc[tdf["street"] == "Interstate 20", "street"] = "I-20"

        # Group by street name and merge geometries
        groupers = ["street", "direction", "lane_count", "urban_name",
                    "urban_cat"]
        grouper = tdf.groupby(groupers)["geometry"]
        segments = grouper.apply(self._merge)
        gdf = gpd.GeoDataFrame(segments, crs="epsg:4326")
        gdf = gdf.reset_index()
        gdf = gdf.to_crs("epsg:5070")

        # Unpack multipolygons
        # sdf = self.unpack_multilines(gdf)

        # Identify intersecting nodes
        # ndf = self.connect_nodes(sdf)

        # Regroup

        # Get the opposite road segments
        odf = self.get_opposite(gdf)
        gdf = gdf[gdf["urban_cat"] != "U"]

        return odf

    @property
    def routes(self):
        """Return the two route vector paths."""
        tdir = os.path.join(
            self.home,
            "data/shapefiles/routes/merged"
        )
        ew = glob(os.path.join(tdir, "*ew*gpkg"))[0]
        ns = glob(os.path.join(tdir, "*ns*gpkg"))[0]
        routes = {"ew": ew, "ns": ns}
        return routes

    def unpack_multilines(self, df):
        """Unpack single multiline entries into multiple line entries."""
        crs = df.crs
        entries = df.apply(self._unpack_multilines, axis=1)
        entries = [e for se in entries for e in se]
        df = gpd.GeoDataFrame(entries)
        df = df.reset_index(drop=True)
        df.crs = crs
        return df

    def _cross_check(self, shape, lines):
        # Could we clip the lines within the shape and check if they cross?
        return shape.intersects(lines)

    def _get_opposite(self, vector1, vector2):
        """Return intersection of vector2 and the bounding box of vector1."""
        from shapely.geometry import box
        bbox = box(*vector1.bounds).buffer(50)  # Trying to catch missing bits
        op_geometry = bbox.intersection(vector2)
        return op_geometry

    def _isstraight(self, segments):
        """Check if a pair of lines are perfectly straight."""
        # Create an ndarray of coordinates
        coords = ops.linemerge(segments).xy
        aline = np.array(coords)

        # Get the first x,y coordinates
        x0, y0 = aline[:, 0]
        x1, y1 = aline[:, 1]

        # Check each coordinate against the first line until a bend is found 
        for i in range(2, aline.shape[1]):
            x, y = aline[:, i]
            if  (x0 - x)  * (y0 - y) != (x1 - x)  * (y1 - y):
                return False

        return True

    def _merge(self, lines):
        """Merge a group of line strings into one."""
        nlines = lines[~pd.isnull(lines)]
        nline = cascaded_union(nlines.values)
        return nline

    def _par_apply(self, data, func, n=None, **kwargs):
        if isinstance(data, gpd.GeoDataFrame):
            def global_fun(args):
                data, func, kwargs = args
                data["geometry"] = data.apply(func, **kwargs, axis=1)
                return data
        else:
            def global_fun(args):
                data, func, kwargs = args
                out = data.apply(func, **kwargs)
                return out

        if n:
            ncpu = n
        else:
            ncpu = os.cpu_count()

        chunks = np.array_split(data, ncpu)
        arg_list = [(cdata, func, kwargs) for cdata in chunks]
        datas = []
        with mp.Pool(ncpu) as pool:
            for cdata in tqdm(pool.imap(global_fun, arg_list), total=ncpu):
                datas.append(cdata)

        data = pd.concat(datas)
        return data

    def _unpack_multilines(self, entry):
        """Return multiple single line entry for a single multiline entry."""
        new_entries = []
        if isinstance(entry["geometry"], MultiLineString):
            for geom in entry["geometry"]:
                new_entry = entry.copy()
                new_entry["geometry"] = geom
                new_entries.append(new_entry)
        else:
            new_entries = [entry]
        return new_entries

    def main(self):
        """Create final merged route table."""
        for direction, file in self.files.items():
            print(f"Merging {file}...")
            df = self.merge(file)
            df = self.buffer_dataset(df)
            dst = self.build_path(direction)
            df.to_file(dst, "GPKG")


if __name__ == "__main__":
    # self = Merger(HOME, DSTDIR)
    merger = Merger(HOME, DSTDIR)
    merger.main()
