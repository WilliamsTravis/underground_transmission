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

from ssurgo import NATSGO, SSURGO, HOME


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

    # Create a nice dictionary from the table variable
    mapping = dict(zip(table["mukey"], table[variable]))
    arrays = np.array_split(array, 100)
    out = []
    for i, array in enumerate(arrays):
        if i == 50:
            break
        array2 = np.vectorize(mapping.get)(array)


if __name__ == "__main__":
    rpath = "../data/ssurgo/gssurgo/gssurgo_co.tif"
    dst = "../data/ssurgo/gssurgo/gssurgo_co_test.tif"
    variable = "brockdepmin"
    targetdir = os.path.join(HOME, "data/ssurgo")
    state = "CO"
    natsgo = NATSGO(targetdir, state)
    ssurgo = SSURGO(targetdir, state)    
    table = ssurgo.table
