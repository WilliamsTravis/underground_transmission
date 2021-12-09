#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 12:47:19 2021

@author: travis
"""
import os

import numba as nb
import numpy as np
import rasterio as rio

from numba import float64, int32, int64
from ssurgo import NATSGO, SSURGO, HOME
from tqdm import tqdm


# @nb.guvectorize([(int32[:], float64[:])], "(m,n),(m,n)")
def mapit(sarray, mapping):
    """Mapping values to an array."""
    for i in range(0, mapping.shape[1] - 1):
        key, value = mapping[:, i:i+1]
        sarray[sarray == key] = value


def to_raster(rpath, table, variable, dst):
    """Map a table's variable to a raster using the mukey."""
    # Read in raster
    r = rio.open(rpath)
    profile = r.profile
    array = r.read(1)    

    # Make sur ethe mukey is numerical
    table["mukey"] = table["mukey"].astype(int)
    table = table[["mukey", variable]].drop_duplicates()
    table = table[~np.isnan(table[variable])]

    # Create a 2D vector as a stand in for a dictionary ("scalars" required)
    mapping = np.vstack([table["mukey"], table[variable]])
    mapdict = dict(zip(table["mukey"], table[variable]))

    # Assign values one section at a time
    arrays = np.array_split(array, 100)
    narrays = []
    for tarray in arrays:
        mapit(tarray, mapping)
        narrays.append(tarray)







if __name__ == "__main__":
    rpath = "../data/ssurgo/gssurgo/gssurgo_co.tif"
    dst = "../data/ssurgo/gssurgo/gssurgo_co_test.tif"
    variable = "brockdepmin"
    targetdir = os.path.join(HOME, "data/ssurgo")
    state = "CO"
    natsgo = NATSGO(targetdir, state)
    ssurgo = SSURGO(targetdir, state)    
    table = ssurgo.table
