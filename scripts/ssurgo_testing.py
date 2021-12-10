#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 12:47:19 2021

@author: travis
"""
import os

import multiprocessing as mp
import numpy as np
import pygeoprocessing as pygp
import rasterio as rio

from osgeo import gdal
from ssurgo import NATSGO, SSURGO, HOME, tile_mukeys
from tqdm import tqdm


def mapit(arg):
    """Mapping values to an raster."""
    # Reclassify
    raster_file, mapdict, variable = arg
    dst = raster_file.replace("mukey", variable)
    pygp.geoprocessing.reclassify_raster(
        base_raster_path_band=(raster_file, 1),
        value_map=mapdict,
        target_raster_path=dst,
        target_datatype=gdal.GDT_Int16,
        target_nodata=-9999,
        values_required=True  # Temporarily off for testing
    )


def to_raster(array, profile, rpath):
    """Write a raster to its rpath."""
    with rio.open(rpath, "w", **profile) as file:
        file.write(array, 1)


def map_variable(raster_file, table, variable, dst):
    """Map a table's variable to a raster using the mukey."""
    # Several adjustments needed to the lookup values
    table["mukey"] = table["mukey"].astype(int)
    table = table[["mukey", variable]].drop_duplicates()
    table[variable][np.isnan(table[variable])] = -9999

    # Create a dictionary of lookup values
    mapdict = dict(zip(table["mukey"], table[variable]))

    # Tile the mukey grid
    out_folder = os.path.dirname(raster_file)
    ncpu = 12
    ntiles = 100
    rpaths = tile_mukeys(raster_file, out_folder, ntiles, ncpu)

    # Map the values to a new raster
    args = [(rpath, mapdict, variable) for rpath in rpaths]
    with mp.Pool(10) as pool:
        for _ in tqdm(pool.imap(mapit, args), total=len(args)):
            pass


if __name__ == "__main__":
    raster_file = ("/home/travis/github/underground_transmission/data/ssurgo/"
                   "gnatsgo/gnatsgo_conus/gnatsgo_conus_mukey.tif")
    dst = ("/home/travis/github/underground_transmission/data/rasters/"
           "gssurgo_conus_test.tif")
    variable = "wtdepannmin"
    targetdir = os.path.join(HOME, "data/ssurgo")
    natsgo = NATSGO(targetdir)
    table = natsgo.table
    map_variable(raster_file, table, variable, dst)

