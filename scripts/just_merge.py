"""Functions to merge a set of HERE segments.

Created on Tue Nov 30 08:31:27 2021

@author: twillia2
"""
import geopandas as gpd
import pandas as pd

from shapely.ops import cascaded_union


KEEPERS = ["FromNodeID", "ToNodeID", "StreetNameBase", "ORDER1_NAME", 
           "UATYP10", "NAME10", "DirOnSign", "LANE_CATEGORY", "geometry"]
COLUMNS = ["from_id", "to_id", "street", "state", "urban_cat", "urban_name",
           "direction", "lane_count", "geometry"]


def merge(self, file):
    """Merge continuous line segments."""
    # Read in and subset table
    df = gpd.read_file(file)
    tdf = df[KEEPERS]
    tdf.columns = COLUMNS

    # Remove major metropolitan urban centers and directionless segments
    tdf = tdf[tdf["direction"] != ""]
    tdf.loc[pd.isnull(tdf["urban_name"]), "urban_name"] = "none"
    tdf = tdf[~tdf["geometry"].isnull()]
1
    # Fix inconsistencies
    tdf.loc[tdf["street"] == "I 40", "street"] = "I-40"
    tdf.loc[tdf["street"] == "I 35", "street"] = "I-35"
    tdf.loc[tdf["street"] == "Ih 35", "street"] = "I-35"
    tdf.loc[tdf["street"] == "Interstate 35", "street"] = "I-35"
    tdf.loc[tdf["street"] == "I 10 Frontage", "street"] = "I-10 Frontage"
    tdf.loc[tdf["street"] == "Interstate 10", "street"] = "I-10"
    tdf.loc[tdf["street"] == "Interstate 20", "street"] = "I-20"

    # Group by street name and merge geometries
    groupers = ["street", "direction", "state", "lane_count", "urban_name",
                "urban_cat"]
    grouper = tdf.groupby(groupers)["geometry"]
    segments = grouper.apply(_merge)
    gdf = gpd.GeoDataFrame(segments, crs="epsg:4326")
    gdf = gdf.reset_index()
    gdf = gdf.to_crs("epsg:5070")

    return gdf


def _merge(lines):
    """Merge a group of line strings into one."""
    nlines = lines[~pd.isnull(lines)]
    nline = cascaded_union(nlines.values)
    return nline
