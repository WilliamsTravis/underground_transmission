"""Create a highway/interstate file from HERE Streets.

Created on Fri Sep 17 09:51:02 2021

@author: twillia2
"""
import os

import fiona
import geopandas as gpd
import numpy as np
import ogr
import pandas as pd
import pathos.multiprocessing as mp

from revruns import rr
from tqdm import tqdm


HOME = "/shared-projects/rev/projects/lpo/fy21/row/"
HERE_FOLDER = "/projects/rev/data/conus/here_streets"
NITERS = 1_000
HWYS = {
    "east_west": ["I-94", "I-90", "I-80", "I-76", "I-70", "US-64", "I-40",
                  "I-24", "I-20", "I-10"],
    "north_south": ["I-5", "I-15", "I-25", "I-29", "I-35", "I-55", "I-65",
                    "I-75", "I-77", "I-85", "I-87", "I-95"]
}
DSTS = {
    "north_south": os.path.join(HOME, "data/ns_interstates.gpkg"),
    "east_west": os.path.join(HOME, "data/ew_interstates.gpkg"),
}



class Highways:
    """Methods for extracting highway/interstate from HERE streets."""

    def __init__(self, home=HOME, here_folder=HERE_FOLDER,
                 route="east_west", highways=HWYS["east_west"]):
        """Initialize Highways object."""
        self.home = rr.Data_Path(home)
        self.here = rr.Data_Path(here_folder)
        self.route = route
        self.highways = highways

    def extract_single(self, gdb):
        """Extract highways from a single state geodatabase."""
        # Create individual state file
        state = self.state(gdb)
        fname = f"{state}_{self.route}.gpkg"
        dst = self.home.join("data/state_hwys", fname, mkdir=True)

        # Read in processed file already saved
        if os.path.exists(dst):
            gdf = gpd.read_file(dst)
        else:
            # Read from original if not saved
            gdf = self.read(gdb)

            # Often an interstate merges with another and becomes the alternate
            query1 = (gdf["StreetNameBase"].isin(self.highways))
            if "StreetNameBaseAlt1" in gdf.columns:
                query2 = (gdf["StreetNameBaseAlt1"].isin(self.highways))
                gdf = gdf[query1 | query2]
            else:
                gdf = gdf[query1]

            # Save if state has any of these interstates
            if gdf.shape[0] > 1:
                gdf.to_file(dst, "GPKG")

        return gdf

    def extract_all(self, dst):
        """Extract highways from all states and merge."""
        gdbs = self.here.contents("*gdb")
        gdfs = []
        for gdb in gdbs:
            print(f"Processing {os.path.basename(gdb)}...")
            gdf = self.extract_single(gdb)
            if gdf.shape[0] > 0:
                gdfs.append(gdf)
        gdf = pd.concat(gdfs).reset_index(drop=True)
        print(f"Writing to {dst}...")
        gdf.to_file(dst, "GPKG")

    @property
    def ncores(self):
        """Return number of cpus available."""
        return mp.cpu_count()

    def nrows(self, gdb):
        """Return row/feature count."""
        ds = ogr.Open(gdb)
        layer = ds.GetLayerByIndex(0)
        nrows = layer.GetFeatureCount()
        del ds
        return nrows

    def read(self, gdb):
        """Read in single file in parallel."""
        def read_wrapper(kwargs):
            return gpd.read_file(**kwargs)

        kwargs = self.read_kwargs(gdb)

        chunks = []
        with mp.Pool(self.ncores) as pool:
            for chunk in tqdm(pool.imap(read_wrapper, kwargs), total=NITERS):
                chunks.append(chunk)

        gdf = pd.concat(chunks).reset_index(drop=True)

        return gdf

    def read_kwargs(self, gdb):
        """Return chunked read key word arguments in gdb file."""
        name = fiona.listlayers(gdb)[0]
        rows = np.arange(0, self.nrows(gdb) + 1, 1)
        row_chunks = np.array_split(rows, NITERS)
        slices = [slice(s.min(), s.max() + 1) for s in row_chunks]
        kwargs = [{"filename": gdb, "layer": name, "rows": s} for s in slices]

        return kwargs

    def state(self, gdb):
        """Return state from gdb file name."""
        name = os.path.basename(gdb).replace(".gdb", "")
        state = name.split("_")[-1].lower()
        return state


if __name__ == "__main__":
    for route in HWYS.keys():
        highways = HWYS[route]
        dst = DSTS[route]
        if not os.path.exists(dst):
            # self = Highways(route=route, highways=highways)
            # break
            extractor = Highways(route=route, highways=highways)
            extractor.extract_all(dst)
