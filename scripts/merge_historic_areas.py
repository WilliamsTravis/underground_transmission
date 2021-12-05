"""Merge Historic Places.

This file is split between several layers: Sites, "distance" (linar features 
like canals, trails, and parkways), Structure, Building, and objects. We want
all of these in one file.

Created on Mon Nov 29 16:15:53 2021

@author: twillia2
"""
import fiona
import geopandas as gpd
import pandas as pd


PATH = "../data/characterizations/NRIS_CR_layers_5070_CONUS.gpkg"
DST = "../data/characterizations/NRIS_CR_layers_merged_5070_CONUS.gpkg"
FIELD = "RESNAME"


def main():
    """Read in file and merge all layers into one."""
    layers = [l for l in fiona.listlayers(PATH) if "py" in l]
    dfs = []
    for layer in layers:
        df = gpd.read_file(PATH, layer=layer)[["geometry", FIELD]]
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_file(DST, "GPKG")


if __name__ == "__main__":
    main()


