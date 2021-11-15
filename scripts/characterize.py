"""Characterize features along routes.

Buffer and characterize files in characterizations folder.


Production Notes:
    - Fix the buffer side issue.
    - How to handle parcels? They're split into individual state files, but
      it wouldn't make sense to add an entire new summary for each state...
      as long as the HERE streets portion is split on states, we can infer the
      state and read in the correct state parcel table.
    - When buffering segments separately, there should be an overlap, handle
      this somehow.


Created on Wed Oct 27 09:54:39 2021

@author: twillia2
"""
import ast
import os
import warnings

from glob import glob

import geopandas as gpd
import numpy as np
import pandas as pd
import pathos.multiprocessing as mp
import shapely

from pandarallel import pandarallel
from rioxarray import rioxarray as xrio
from shapely.geometry import box, LineString, MultiPolygon, MultiLineString
from shapely.geometry import Polygon
from shapely.geometry.collection import GeometryCollection
from shapely.ops import cascaded_union
from shapely.errors import ShapelyDeprecationWarning
from tqdm import tqdm

tqdm.pandas()
pandarallel.initialize(progress_bar=True)
warnings.simplefilter("ignore", ShapelyDeprecationWarning)


HOME = "/shared-projects/rev/projects/lpo/fy21/underground_transmission"
INPUTS = os.path.join(HOME, "data", "tables", "data_sources.xlsx")
FTYPES = {
    "geojson": "vector",
    "gpkg": "vector",
    "gdb": "vector",
    "shp": "vector",
    "tif": "raster"
}
PROBLEM_SIDS = {
    "ns": [9, 22, 24]
}



def find_dirs(line):
    """Find the direction of coordinates in a line."""
    dx = np.diff(line.xy[0])
    dy = np.diff(line.xy[1])
    xdir = int(sum(dx) / abs(sum(dx)))
    ydir = int(sum(dy) / sum(abs(dy)))
    return xdir, ydir


def vbuffer(line, direction="N", sign="+", distance=65, width=4):
    """Visualize the different buffering options we're using."""
    offset = (width / 2) + (distance / 2)

    if sign == "-":
        offset = -offset

    sline = line.parallel_offset(offset)
    road = line.buffer((width / 2))
    buff = sline.buffer(distance / 2)

    full = line.buffer(distance)
    phalf = line.buffer(distance, single_sided=True)
    nhalf = line.buffer(-distance, single_sided=True)

    # Final buffer?
    half = line.buffer((distance / 2), single_sided=True)
    gdata = {
        "geometry": [
            line, 
            sline,
            road,
            buff
        ],
        "value": [
            "center line",
            "offset line",
            "road",
            "buffer"
        ]
    }

    gdf = gpd.GeoDataFrame(gdata)
    ax = gdf.plot(column="value", categorical=True, legend=True)


class Summarize:
    """Methods to use for the various summarizations needed."""

    def __init__(self, home=HOME, inputs_path=INPUTS):
        """Initialize Summarize object."""
        self.home = home
        self.inputs_path = inputs_path
        self.data_dir = os.path.join(home, "data/characterizations")

    def get_dtype(self, dset):
        """Return the type (raster or vector) of a dataset."""
        entry = self.get_entry(dset)
        fpath = entry["Formatted Path"]
        ftype = os.path.splitext(fpath)[-1].replace(".", "").rstrip()
        return FTYPES[ftype]

    def get_entry(self, dset):
        """Return the meta data for a single dataset entry."""
        return self.inputs[self.inputs["Data Name"] == dset].iloc[0]

    def get_function(self, function):
        """Return a method described by its name."""
        return Summarize.__dict__[function]

    def get_key(self, dset, method):
        """Return a key for a given data set and summarization method."""
        dkey = "_".join(dset.lower().split())
        mkey = method.__name__
        return f"{dkey}_{mkey}"

    def get_methods(self, dset):
        """Return the requested summary methods for a dataset."""
        # Unpack info for dataset
        entry = self.get_entry(dset)
        method_names = entry["Summarization"].split(",")

        # Figure out which method to use
        methods = []
        for method in method_names:
            dtype = self.get_dtype(dset)
            method = "_".join(method.split())
            method_key = f"{dtype}_{method}"
            method = self.get_function(method_key)
            methods.append(method)
        return methods

    @property
    def inputs(self):
        """Return input data sheet."""
        df = pd.read_excel(self.inputs_path, sheet_name="data")
        df = df[["Data Name", "Formatted Path", "Summarization", "Category",
                 "Fields", "Legend"]]
        df["Fields"][pd.isnull(df["Fields"])] = "NA"
        df["Legend"][pd.isnull(df["Legend"])] = "NA"
        df = df.dropna().reset_index(drop=True)
        return df

    def raster_mean(self, df, dset):
        """Return the mean value of a raster."""

    def raster_area(self, df, dset):
        """Return the area of pixels for a raster value."""

    def raster_area_by_class(self, df, dset):
        """Return the area of pixels for each raster value."""
        # Get geometric info
        entry = self.get_entry(dset)
        fpath = entry["Formatted Path"]
        with xrio.open_rasterio(fpath) as ds:
            transform = ds.rio.transform()
        res = transform[0]

        # Build a dictionary of areas
        areas = {}
        values = np.unique(df)
        for value in values:
            varray = df[df == value]
            count = varray.shape[0]
            area = count * (res ** 2)
            areas[value] = area

        return areas

    def raster_count(self, df, dset):
        """Return the count of pixels for a raster value."""

    def raster_count_by_class(self, df, dset):
        """Return the count of pixels for each raster value."""

    def vector_area_by_class(self, df, dset):
        """Return the area for each vector value."""

    def vector_count_by_class(self, df, dset):
        """Return the count for each vector value."""

    def vector_count_of_intersections(self, df, dset):
        """Return the count of line, vector intersections."""

    def vector_count_of_polygons(self, df, dset):
        """Return the number of polygons in a vector."""

    def summarize(self, row, dset):
        """Summarize a given dataset within a given extent.

        Parameters
        ----------
        row : shapely.geometry.MultiPolygon
            A buffered section of a route.
        dset : str
            The name of a dataset in the inputs tables.

        Returns
        -------
        dictionary
            A dictionary with a key representing the dataset and method used,
            and the values representing the results of that method.
        """
        # Retrieve subsetted dataset
        df = self._subset(row, dset)

        # Retrieve the appropriate methods for this dataset
        methods = self.get_methods(dset)

        # For each method retrieve a value and catalog it
        summary = {}
        for method in methods:
            out = method(self, df, dset)

            # We need a key for the method and dataset
            key = self.get_key(dset, method)
            summary[key] = out

        return summary

    def _subset(self, row, dset):
        """Return a subset of the dataset within the row."""
        # Unpack data elements
        entry = self.get_entry(dset)
        fpath = entry["Formatted Path"]

        # Let's not break the whole thing if the path is missing
        if os.path.exists(fpath):

            # Create a bounding box in which to read in dataset
            bbox = row.bounds
    
            # Use the appropriate method for reading in dataset
            if self.get_dtype(dset) == "vector":
                df = gpd.read_file(fpath, bbox=bbox)
    
            elif self.get_dtype(dset) == "raster":
                df = self._subset_raster(row, dset)
        else:
            print(f"{fpath} does not exist, skipping...")
            df = None

        return df

    def _subset_raster(self, row, dset):
        """Subset a raster with an row geometry."""
        # Unpath dataset information
        fpath = self.get_entry(dset)["Formatted Path"]

        # Create a bounding box in which to read in dataset
        bbox = row.bounds

        # Create a mask out of the row
        with xrio.open_rasterio(fpath) as ds:
            df = ds.rio.clip_box(*bbox)
            df = df.rio.clip(row)

        return df.data[0]


class Characterize(Summarize):
    """Methods for characterizing ROW around line segments."""

    def __init__(self, home=HOME, inputs_path=INPUTS, distance=65):
        """Initialize Characterize object."""
        super().__init__(home, inputs_path)
        self.distance = distance

    def __repr__(self):
        """Return Characterize object representation string."""
        attrs = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
        msg = f"<Characterize object: {attrs}>"
        return msg

    def characterize(self, direction):
        """Buffer and characterize one route table."""
        # Buffer or retrieve buffer
        print(f"Characterizing {direction}...")
        subdir = os.path.join(self.home, "data/shapefiles/routes/buffered")
        tpath = f"{direction}_interstates_buffered.gpkg"
        os.makedirs(subdir, exist_ok=True)
        buffered_dst = os.path.join(subdir, tpath)
        if os.path.exists(buffered_dst):
            df = gpd.read_file(buffered_dst)
        else:
            df = self.buffer_dataset(direction)
            df.to_file(buffered_dst, "GPKG")

        # Loop through each segment - running serially for now
        output = []
        for s_i, s_entry in tqdm(df.iterrows(), total=df.shape[0]):
        
            # Extract the buffer
            row = s_entry["geometry"]
        
            # Loop through characterization dataset inputs
            for i_i, i_entry in self.inputs.iterrows():
        
                # Get dataset, check if we're using parcels
                dset = i_entry["Data Name"]
                if "parcels" in dset:
                    self.find_parcel()
        
                # Get summary for all characterization methods
                summaries = self.summarize(row, dset)
        
                # The summaries can be a value or dict of values
                for method, summary in summaries.items():
                    if not isinstance(summary, dict):
                        s_entry[method] = summary
                    else:
                        for cat, value in summary.items():
                            # This key is important here (no units column)
                            cat = self.get_cat_lookup(cat, dset)
                            if self.get_units(cat):
                                cat = cat + " " + self.get_units(cat)
                            if cat:
                                s_entry[cat] = value
                    output.append(s_entry)
        
        df = pd.concat(output)
        return df

    def clip_bbox(self, vector1, vector2):
        """Clip a vector within the bounding box of another."""
        bbox = box(*vector1.bounds)
        # bbox = bbox.buffer(100)
        return bbox.intersection(vector2)

    def find_side(self, line, other_line, direction):
        """Determine the correct side to offset a road."""
        # The direction of coordinates sets the offset side
        xdir, ydir = find_dirs(line)

        # This might change based on the direction, and we might need ydir
        # if direction == "N":

        if xdir == -1 and ydir == 1:
            side = "right"
        else:
            side = "left"

        return side

    def fix_line(self, line):
        """Make everyline flow in the same direction."""
        xdir, _ = find_dirs(line)
        if xdir == -1:
            xy = line.xy
            xs = xy[0][::-1]
            ys = xy[1][::-1]
            coords = [[xs[i], ys[i]] for i in range(len(xs))]
            line = LineString(coords)
        return line

    def get_cat_lookup(self, value, dset):
        """Get lookup values for categorical datasets."""
        entry = self.get_entry(dset)
        legend = ast.literal_eval(entry["Legend"])
        if legend:
            if value in legend:    
                cat = legend[value]
                cat = cat.lower().replace(",", "").replace("/", " ")
                cat = "_".join(cat.split())
            else:
                cat = None
        else:
            cat = value

        return cat

    def get_units(self, cat):
        """Get the units associated with a method."""
        units = None
        if cat:
            if "area" in cat:
                units = "m2"
        return units

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

    def _cross_check(self, shape, lines):
        # Could we clip the lines within the shape and check if they cross?
        return shape.intersects(lines)

    def _par_apply(self, df, func, **kwargs):
        if isinstance(df, gpd.GeoDataFrame):
            def global_fun(args):
                df, func, kwargs = args
                df["geometry"] = df.apply(func, **kwargs, axis=1)
                return df
        else:
            def global_fun(args):
                df, func, kwargs = args
                out = df.apply(func, **kwargs)
                return out

        ncpu = os.cpu_count()
        chunks = np.array_split(df, ncpu)
        arg_list = [(cdf, func, kwargs) for cdf in chunks]
        dfs = []
        with mp.Pool(ncpu) as pool:
            for cdf in tqdm(pool.imap(global_fun, arg_list), total=ncpu):
                dfs.append(cdf)

        ndf = pd.concat(dfs)
        return ndf

    def main(self, buffer=65):
        """Buffer and characterize all segements."""
        # Read route geodatabase with route 'segments'
        for direction, _ in self.routes.items():
            df = self.characterize(direction)


if __name__ == "__main__":
    self = Characterize(HOME)
