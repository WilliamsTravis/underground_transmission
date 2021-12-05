"""Characterize features along routes.

Buffer and characterize files in characterizations folder.

Production Notes:
    - How to handle parcels? They're split into individual state files, but
      it wouldn't make sense to add an entire new summary for each state...
      as long as the HERE streets portion is split on states, we can infer the
      state and read in the correct state parcel table.


Created on Wed Oct 27 09:54:39 2021

@author: twillia2
"""
import json
import os
import warnings

from glob import glob

import geopandas as gpd
import numpy as np
import pandas as pd
import pathos.multiprocessing as mp

from merge_segments import cplot
from rioxarray import rioxarray as xrio
from shapely.geometry import box
from shapely.errors import ShapelyDeprecationWarning
from tqdm import tqdm

tqdm.pandas()
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


def find_row(df, i_entry):
    """Little function to find the first intersecting right-of-way."""
    summarizer = Summarize()
    dset = i_entry["Data Name"]
    for row in df["geometry"]:
        out = summarizer._subset(row, dset)
        if out.shape[0] > 0:
            return row


class Summarize:
    """Methods to use for the various summarizations needed."""

    def __init__(self, home=HOME, inputs_path=INPUTS):
        """Initialize Summarize object."""
        self.home = home
        self.inputs_path = inputs_path
        self.data_dir = os.path.join(home, "data/characterizations")
        self.crs = "epsg:5070"  # Hardcoding for time

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
        dkey = dset.replace("-", "").replace("(", "").replace(")", "")
        dkey = dkey.replace(",", " ")
        dkey = "_".join(dkey.lower().split())
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
        # Read in raw data sheet
        df = pd.read_excel(self.inputs_path, sheet_name="data")
        df = df[["Data Name", "Formatted Path", "Summarization", "Category",
                 "Fields", "Legend"]]

        # Keep only entries with necessary elements
        nonans = ["Formatted Path", "Summarization"]
        df = df.dropna(subset=nonans).reset_index(drop=True)
        cmethods = df["Summarization"].str.contains("class")
        nofields = df["Fields"].isnull()
        df = df[~(cmethods & nofields)]

        # Not ready for folders or geodatabases yet
        exts = df["Formatted Path"].apply(lambda x: os.path.splitext(x)[-1]) # Temporary
        df = df[exts != '']

        # Trim any whitespace in the file paths
        df["Formatted Path"] = df["Formatted Path"].apply(lambda x: x.strip())

        return df

    def parcel_file(self, s_entry):
        """Find the right parcel state."""
        state = s_entry["state"]
        state = state.lower()
        state = "_".join(state.split())
        pattern = f"/projects/rev/data/conus/lightbox/*{state}*5070.gpkg"
        file = glob(pattern)[0]
        return file

    def raster_mean(self, sdf, dset):
        """Return the mean value of a raster."""

    def raster_area(self, sdf, dset):
        """Return the area of pixels for a raster value."""

    def raster_area_by_class(self, sdf, dset):
        """Return the area of pixels for each raster value."""
        # Get geometric info
        entry = self.get_entry(dset)
        fpath = entry["Formatted Path"]
        with xrio.open_rasterio(fpath) as ds:
            transform = ds.rio.transform()
        res = transform[0]

        # Build a dictionary of areas
        areas = {}
        values = np.unique(sdf)
        for value in values:
            varray = sdf[sdf == value]
            count = varray.shape[0]
            area = count * (res ** 2)
            areas[value] = area

        return areas

    def raster_count(self, sdf, dset):
        """Return the count of pixels for a raster value."""

    def raster_count_by_class(self, sdf, dset):
        """Return the count of pixels for each raster value."""

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
        # Retrieve subsetted dataset and appropriate methods for this dataset
        sdf = self._subset(row, dset)
        methods = self.get_methods(dset)

        # For each method retrieve a value and catalog it
        summary = {}
        for method in methods:
            out = method(self, sdf, dset)
            key = self.get_key(dset, method)
            summary[key] = out

        return summary

    def vector_area_by_class(self, sdf, dset):
        """Return the area for each vector value."""
        values = {}        
        field = self.get_entry(dset)["Fields"]
        for sub_field in sdf[field].unique():
            fdf = sdf[sdf[field] == sub_field]
            area = fdf["geometry"].area.sum()
            values[sub_field] = area
        return values

    def vector_count_by_class(self, sdf, dset):
        """Return the count for each vector value."""
        values = {}        
        field = self.get_entry(dset)["Fields"]
        for sub_field in sdf[field].unique():
            fdf = sdf[sdf[field] == sub_field]
            count = fdf["geometry"].shape[0]
            values[sub_field] = count
        return values

    def vector_count_of_intersections(self, sdf, dset):
        """Return the count of line, vector intersections."""
        return sdf.shape[0]  # Same as below

    def vector_count_of_polygons(self, sdf, dset):
        """Return the number of polygons in a vector."""
        return sdf.shape[0]

    def vector_mean(self, sdf, dset):
        """Return the average value of a set of vectors."""
        field = self.get_entry(dset)["Fields"]
        sdf["area"] = sdf["geometry"].area
        return np.average(sdf[field], weights=sdf["area"])

    def _subset(self, row, dset):
        """Return a subset of the dataset within the row."""
        # Unpack data elements
        entry = self.get_entry(dset)
        fpath = entry["Formatted Path"]

        # Let's not break the whole thing if the path is missing
        if os.path.exists(fpath):

            # Use the appropriate method for reading in dataset
            if self.get_dtype(dset) == "vector":
                sdf = self._subset_vector(row, dset)

            elif self.get_dtype(dset) == "raster":
                sdf = self._subset_raster(row, dset)
        else:
            print(f"{fpath} does not exist, skipping...")
            sdf = None

        return sdf

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

        return df.data[0]  # 0 because this is the 1st layer in a 3D set

    def _subset_vector(self, row, dset):
        """Subset a raster with a row geometry."""
        # Unpath dataset information
        fpath = self.get_entry(dset)["Formatted Path"]
        field = self.get_entry(dset)["Fields"]

        # Create a bounding box in which to read in dataset
        bbox = row.bounds

        # Subset for bounding box and then clip by row
        sdf = gpd.read_file(fpath, bbox=bbox)  # is usecols not a thing? This might not be the best way to do this
        if field is not np.nan:
            sdf = sdf[["geometry", field]]
        else:
            sdf = sdf[["geometry"]]
        
        sdf = gpd.clip(sdf, row)

        return sdf

    def _read_vector(self, fpath, cols=None, bbox=None):
        """Read in specific columns and bounding box of a geodataframe."""
        # Get the columns to skip
        all_cols = self._cols(fpath)
        if cols:
            skip_cols = list(set(all_cols) - set(cols))
        else:
            skip_cols = list(set(all_cols) - set(all_cols[0]))

        # Read in with or without bounding box
        if bbox:
            gdf = gpd.read_file(fpath, ignore_fields=skip_cols, driver="GeoJSON")

    def _cols(self, fpath):
        """Return columns of shapefile.
        
        Currently only works for geojsons, but keeping in case we need it.
        """
        # Open raw data file (quickest way)
        r = open(fpath).readlines()

        # Dig an entry out of this mess, anyone after 4 or 5 will do
        entry = r[10]

        # Start at properties, end at the next closing bracket
        start = '"properties": '
        end = "}"
        entry = entry[entry.index(start) + len(start): ]
        entry = entry[:entry.index(end) + len(end)]

        # This is a json string, load that in as an object
        entry = json.loads(entry)
        cols = list(entry.keys())

        return cols
    

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
        # Read in buffers
        print(f"Characterizing {direction} interstates...")
        fpath = self.routes[direction]
        df = gpd.read_file(fpath)
        crs = df.crs

        # Loop through each segment - running serially for now
        output = df.progress_apply(self._characterize, axis=1)
        # output = self._par_apply(df, self._characterize)
        df = gpd.GeoDataFrame(output, crs=crs)

        return df

    def clip_bbox(self, vector1, vector2):
        """Clip a vector within the bounding box of another."""
        bbox = box(*vector1.bounds)
        return bbox.intersection(vector2)

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
            "data/shapefiles/routes/buffered"
        )
        ew = glob(os.path.join(tdir, "*ew*gpkg"))[0]
        ns = glob(os.path.join(tdir, "*ns*gpkg"))[0]
        routes = {"ew": ew, "ns": ns}
        return routes

    def _characterize(self, s_entry):
        # Single entry characterization
        row = s_entry["geometry"]
        for i_i, i_entry in self.inputs.iterrows():
            # Get dataset, check if we're using parcels
            dset = i_entry["Data Name"]

            if "parcels" in dset:
                break
                summaries = self.parcel_distance(row, s_entry)

            # Get summary for all characterization methods
            try:
                summaries = self.summarize(row, dset)
            except:
                road = s_entry["street"]
                sid = s_entry["sid"]
                print("Exception on interstate {road}, sid {sid}, dataset {dset}")
                raise

            # The summaries can be a value or dict of values
            for col, summary in summaries.items():
                s_entry[col] = summary

        return s_entry

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
            fpath = f"{direction}_characterizations.gpkg"
            dst = os.path.join(self.data_dir, "final", fpath)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            df.to_file(dst, "GPKG")


if __name__ == "__main__":
    self = Characterize(HOME)
    # characterizer = Characterize(HOME)
    # characterizer.main()
