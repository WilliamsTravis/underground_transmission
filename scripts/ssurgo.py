"""Retrieve and format SSURGO.

Notes
-----
Variable descriptions:
    https://data.nal.usda.gov/system/files/SSURGO_Metadata_-_Table_Column_Descriptions.pdf#page=81

Units:
    https://jneme910.github.io/CART/chapters/Soil_Propert_List_and_Definition


Created on Mon Oct 11 09:46:55 2021

@author: travis
"""
import os
import warnings
import zipfile

from functools import lru_cache
from glob import glob

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import wget

from requests import HTTPError

warnings.filterwarnings("ignore")


URL = ("https://websoilsurvey.sc.egov.usda.gov/DSD/Download/Cache/STATSGO2/"
       "wss_gsmsoil_US_[2016-10-16].zip")
GSSURGO = ("https://nrcs.app.box.com/v/soils/")
US = "https://www2.census.gov/geo/tiger/TIGER2017//STATE/tl_2017_us_state.zip"


class Download:
    """Methods for downloading the gSSURGO files. Will require BOX SDK."""

    def __init__(self, state="CO"):
        """Initialize Download object."""
        self.state = state

    @property
    @lru_cache()
    def state_url(self):
        """Return the url for just one state gSSURGO file."""
        def noparser(content, target="gSSURGO by State"):
            """It's too much to parse a java script enabled html request."""
            idx = content.index(target)
            lcontent = content[idx::-1]
            rcontent = lcontent[:lcontent.index(':"di"')][::-1]
            fid = rcontent[:rcontent.index(",")]
            return fid
    
        # Find the state gssurgo page id
        with requests.get(GSSURGO) as r:
            content = r.text
            fid = noparser(content, target="gSSURGO by State")
    
        # Find the specific state file id
        target=f"gSSURGO_{self.state}.zip"
        sfid = None
        for page in [1, 2, 3]:
            url = os.path.join(GSSURGO, f"folder/{fid}?page={page}")
            try:
                with requests.get(url) as r:
                    content = r.text
                    sfid = noparser(content, target=target)
            except ValueError:
                pass

        if not sfid:
            raise ValueError(f"Could not find a file for {self.state}")

        # Build the url
        url = os.path.join(GSSURGO, f"file/{sfid}")
    
        return url

    def get(self):
        """Get the SSURGO dataset."""
        # Get just one state if provided
        if self.state:
            url = self.state_url

        dst = os.path.join(self.trgtdir, f"gSSURGO_{self.state}.zip")
        if not os.path.exists(dst):
            try:
                wget.download(url, dst)
            except:
                raise HTTPError(f"{URL} could not be found, perhaps the "
                                "dataset has been updated. Try updating "
                                "the URL constant in this script.")
            with zipfile.ZipFile(dst, "r") as zf:
                zf.extractall(path=self.trgtdir)



class SSURGO:
    """Methods for formatting the gSSURGO dataset to return vectors."""

    def __init__(self, gdb, shapefile):
        """Initialize SSURGO object."""
        self.gdb = gdb
        self.trgtdir = os.path.dirname(self.gdb)
        self.shapefile = shapefile

    def __repr__(self):
        """Return representation string."""
        attrs = []
        for key, value in self.__dict__.items():
            attrs.append(f"{key}={value}")
        attrs = ", ".join(attrs)
        return f"<SSURGO object: {attrs}>"

    def build(self, variable):
        """Build a useable dataset out of SSURGO."""
        # Get the component table (may need to add tables for certain keys)
        table = self.table.copy()

        # Let's keep the horizon information
        keepers = ["mukey", "hzname", "hzdept_r", "hzdepb_r"]
        if variable not in keepers:
            table = table[keepers + [variable]]
        else:
            table = table[keepers]

        # Only keep the entries with values for our target variable
        table = table[table[variable].notna()]

        # Additionally, this may be horizon independent
        grouper = table.groupby("mukey", as_index=False)[variable]
        table["nvar"] = grouper.transform(pd.Series.nunique)
        if table["nvar"].max() == 1:
            table = table[["mukey", variable]].drop_duplicates()

        # Merge with our US shapefile
        shape = self.shape
        df = pd.merge(shape, table, on="mukey")

        return df

    @property
    def chaashto(self):
        """Return the aashtocl text file as a table."""
        pattern = os.path.join(self.trgtdir, "**/*chaashto.txt")
        path = glob(pattern, recursive=True)[0]
        df = pd.read_csv(path, sep="|", header=None)
        df.columns = self._cols("chaashto")
        return df

    @property
    def cgeomordesc(self):
        """Return the cogeomordesc text file as a table."""
        pattern = os.path.join(self.trgtdir, "**/*cgeomord.txt")
        path = glob(pattern, recursive=True)[0]
        df = pd.read_csv(path, sep="|", header=None)
        df.columns = self._cols("cogeomordesc")
        return df

    @property
    def chorizon(self):
        """Return the chorizon text file as a table."""
        pattern = os.path.join(self.trgtdir, "**/*chorizon.txt")
        path = glob(pattern, recursive=True)[0]
        df = pd.read_csv(path, sep="|", header=None)
        df.columns = self._cols("chorizon")
        return df

    @property
    def components(self):
        """Return the components text file as a table."""
        pattern = os.path.join(self.trgtdir, "**/*comp.txt")
        path = glob(pattern, recursive=True)[0]
        df = pd.read_csv(path, sep="|", header=None)
        df.columns = self._cols("component")
        return df

    @property
    def layers(self):
        """Return a list of avialabe variables."""
        layers = list(fiona.listlayers(self.gdb))
        layers.sort()
        return layers

    @property
    def mapunit(self):
        """Return the mapunit text file as a table."""
        pattern = os.path.join(self.trgtdir, "**/*mapunit.txt")
        path = glob(pattern, recursive=True)[0]
        df = pd.read_csv(path, sep="|", header=None)
        df = df[df[0].str.contains("s")]
        return df

    @property
    def muaggatt(self):
        """Return the chorizon text file as a table."""
        pattern = os.path.join(self.trgtdir, "**/*muaggatt.txt")
        path = glob(pattern, recursive=True)[0]
        df = pd.read_csv(path, sep="|", header=None)
        df = df[df[0].str.contains("s")]
        df.columns = self._cols("muaggatt")
        return df

    @property
    @lru_cache()
    def shape(self):
        """Return the original geodatabase just for CONUS."""
        # Get the ssurgo map unit layer
        ssurgo = gpd.read_file(self.shapefile)
        ssurgo.columns = [c.lower() for c in ssurgo.columns]

        # Let's add the state name to each
        centroids = gpd.GeoDataFrame({"geometry": ssurgo["geometry"].centroid})
        us = gpd.read_file(US)
        us.crs = "epsg:4269"
        us = us.to_crs("epsg:4326")
        us = us[["geometry", "NAME"]]
        us = us.rename({"NAME": "state"}, axis=1)
        slu = gpd.sjoin(us, centroids)[["state", "index_right"]]
        smap = dict(zip(slu["index_right"], slu["state"]))
        ssurgo["state"] = ssurgo.index.map(smap)

        return ssurgo

    @property
    @lru_cache()
    def table(self):
        """Return the components table."""
        table = pd.merge(self.chorizon, self.components, on="cokey")
        table = pd.merge(table, self.muaggatt, on="mukey")
        table = pd.merge(table, self.chaashto, on="chkey")
        table["mukey"] = table["mukey"].astype(str)
        return table

    @property
    def variables(self):
        """Return a list of available variables."""
        variables = list(self.table.keys())
        variables.sort()
        return variables

    def _cols(self, layer):
        """Return the column names (without geometry) of a gdb layer."""
        cols = list(gpd.read_file(self.gdb, layer=layer, rows=0).columns)
        cols.remove("geometry")
        return cols


if __name__ == "__main__":
    gdb = ("/home/travis/github/underground_transmission/data/ssurgo/"
           "gSSURGO_WY.gdb")
    dbf = ("/home/travis/github/underground_transmission/data/ssurgo/"
           "wss_gsmsoil_US_[2016-10-13]/spatial/gsmsoilmu_a_us.shp")
    variable = "brockdepmin"
    self = SSURGO(gdb, dbf)
    
    