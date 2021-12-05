"""Read in FEMA Flood Hazard, select just the category column, save to gpkg.

Created on Tue Nov 30 12:21:12 2021

@author: twillia2
"""
import fiona
import geopandas as gpd


PATH = ("/shared-projects/rev/projects/lpo/fy21/underground_transmission/"
        "data/shapefiles/fema/NFHL_Key_Layers.gdb")
COLUMNS = ["geometry", "FLD_SUBTY"]


def getit(**kwargs):
    with fiona.open(PATH, **kwargs) as source:
        for feature in source:
            f = {k: feature[k] for k in ['id', 'geometry']}
            f['properties'] = {k: feature['properties'][k] for k in COLUMNS}
            yield f


def main():
    out = getit()
    df = gpd.GeoDataFrame.from_features(out, ['prop1', 'prop2'])

