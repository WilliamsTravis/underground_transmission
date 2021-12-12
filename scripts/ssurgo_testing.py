#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 12:47:19 2021

@author: travis
"""
import json
import os

import multiprocessing as mp
import numpy as np
import rasterio as rio

from ssurgo import NATSGO, HOME, tile_mukeys


MUKEY_GRID = ("/home/travis/github/underground_transmission/data/ssurgo/"
               "gnatsgo/gnatsgo_conus/mukey_grid.tif")
NAVALUE = 65535
TARGET_DIR = os.path.join(HOME, "data/ssurgo")


def mapit(arg):
    """Mapping values to an raster."""
    # Read in raster and its profile
    raster_file, mapdict, variable, dst = arg
    r = rio.open(raster_file)
    profile = r.profile
    array = r.read(1)

    # There might be some missing values
    uarray = np.unique(array)
    missing = [uv for uv in uarray if uv not in mapdict]
    for miss in missing:
        mapdict[miss] = NAVALUE

    # And this is the thing that does it
    array = np.vectorize(mapdict.get)(array)

    # Clean up array
    array[array == None] = NAVALUE
    array = array.astype("uint16")

    # Reset no data and write to file
    profile["nodata"] = NAVALUE
    to_raster(array, profile, dst)
    del array


def to_raster(array, profile, rpath):
    """Write a raster to its rpath."""
    with rio.open(rpath, "w", **profile) as file:
        file.write(array, 1)


def build_dict(variable):
    """Build lookup dictionary for mukeys to variable values."""
    dst = os.path.join(HOME, f"data/ssurgo/lookups/{variable}.json")
    if os.path.exists(dst):
        with open(dst, "r") as file:
            mapdict = json.load(file)
        mapdict = {int(k): v for k, v in mapdict.items()}
    else:
        natsgo = NATSGO(TARGET_DIR)
        table = natsgo.table
        table = table[["mukey", variable]].drop_duplicates()
        table[variable][np.isnan(table[variable])] = NAVALUE
        mukeys = [x.item() for x in table["mukey"].values]
        values = table[variable].values

        # Create a dictionary of lookup values
        mapdict = dict(zip(mukeys, values))
        del table

        # Now, if there are any missing values, add them in as max uint16
        keymin = min(mapdict.keys())
        keymax = max(mapdict.keys())
        all_keys = np.arange(keymin, keymax + 1, 1)
        all_keys = [k.item() for k in all_keys]
        for key in all_keys:
            if key not in mapdict.keys():
                mapdict[key] = NAVALUE
        del all_keys

        # We need a key for the nan value
        mapdict[np.iinfo("uint32").max] = NAVALUE

        # And save
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as file:
            file.write(json.dumps(mapdict))

    return mapdict


def map_variable(mukey_path, mapdict, variable):
    """Map a table's variable to a raster using the mukey."""
     # Tile the mukey grid
    out_folder = os.path.dirname(mukey_path)
    ncpu = mp.cpu_count() - 1
    ntiles = 100
    rpaths = tile_mukeys(mukey_path, out_folder, ntiles, ncpu)

    # Create arguments for unprocessed files
    args = []
    out_paths = []
    for rpath in rpaths:
        dst = rpath.replace("mukey", variable)
        out_paths.append(dst)
        if not os.path.exists(dst):
            args.append((rpath, mapdict, variable, dst))

    # Map the values to new rasters
    with mp.Pool(ncpu) as pool:
        pool.map(mapit, args)
    pool.join()

    return out_paths


def main(variable, dst):
    # Get the path to the mukey grid and build the lookup dictionary
    print(f"Building raster for {variable}...")
    mukey_path = MUKEY_GRID
    mapdict = build_dict(variable)

    # Map values to keys in tiles
    print("Mapping values to tiled mukey grid...")
    files = map_variable(mukey_path, mapdict, variable)

    # Merge tiles
    print(f"Merging tiles and saving final layer to {dst}...")
    options = ["tiled=yes", "compress=lzw", "blockxsize=256", "blockysize=256"]
    cos = ["-co " + co for co in options]
    cmd = " ".join(["gdal_merge.py", "-n", str(NAVALUE), "-a_nodata",
                    str(NAVALUE), *cos, *files, "-o", dst])
    os.system(cmd)


if __name__ == "__main__":
    for variable in ["wtdepannmin", "brockdepmin"]:
        dst = ("/home/travis/github/underground_transmission/data/rasters/"
               f"gssurgo_conus_{variable}.tif")
        main(variable, dst)
