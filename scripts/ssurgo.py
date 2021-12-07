"""Retrieve and format SSURGO.

Notes
-----
Variable descriptions:
    https://data.nal.usda.gov/system/files/SSURGO_Metadata_-_Table_Column_Descriptions.pdf#page=81

Units:
    https://jneme910.github.io/CART/chapters/Soil_Propert_List_and_Definition

I could not get the Download methods to work for gSSURGO, they're buried in a
NRCS Box account and I can't figure how the API works with this. So, this
script assumes that you've downloaded and unzipped gSSURGO and gNATSGO state
files here:
    ../data/ssurgo/gssurgo & ../data/ssurgo/gnatsgo

Created on Mon Oct 11 09:46:55 2021

@author: travis
"""
import os
import time
import warnings
import zipfile

from functools import lru_cache
from glob import glob

import dask_geopandas as dgpd
import fiona
import geopandas as gpd
import pandas as pd
import requests
import us
import wget

from pandarallel import pandarallel as pdl
from requests import HTTPError
from tqdm import tqdm

pdl.initialize(progress_bar=True, use_memory_fs=True)
tqdm.pandas()
warnings.filterwarnings("ignore")


HOME = os.path.expanduser("~/github/underground_transmission")
URL = ("https://websoilsurvey.sc.egov.usda.gov/DSD/Download/Cache/STATSGO2/"
       "wss_gsmsoil_US_[2016-10-16].zip")
GSSURGO = ("https://nrcs.app.box.com/v/soils/")
US = "https://www2.census.gov/geo/tiger/TIGER2017//STATE/tl_2017_us_state.zip"
STATES = [s.abbr for s in us.STATES if s.abbr not in ["HI", "AK"]]


def diff(g1, g2):
    """Return the difference between geometry 1 and geometry 2."""
    try:
        g = g1.difference(g2)
    except:
        raise
    return g


class Download:
    """Methods for downloading the gSSURGO files. Will require BOX SDK."""

    def __init__(self, targetdir, state="CO"):
        """Initialize Download object."""
        self.state = state
        self.targetdir = os.path.expanduser(os.path.abspath(targetdir))
        os.makedirs(self.targetdir, exist_ok=True)

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
        target = f"gSSURGO_{self.state}.zip"
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

    def download(self, url):
        """Piece together needed request address and download."""
        with requests.get(url) as r:
            content = r.content.decode()
        apihost = self._unpack_address(content, pattern="apiHost")
        apikey = self._unpack_address(content, pattern="accountHost")
        dlurl = self._unpack_address(content, "Download")

    def _unpack_adress(self, content, pattern):
        """Find the address associated with a pattern."""
        address = content[content.index(pattern)- 25:]
        address = address.replace(pattern + '":"', "")
        address = address[:address.index('","')]
        address = address.replace("\\", "")
        return address

    def get(self):
        """Get the SSURGO dataset."""
        # Get just one state if provided
        url = self.state_url
        dst = os.path.join(self.targetdir, f"gSSURGO_{self.state}.zip")
        if not os.path.exists(dst):
            try:
                wget.download(url, dst)
            except:
                raise HTTPError(f"{URL} could not be found, perhaps the "
                                "dataset has been updated. Try updating "
                                "the URL constant in this script.")
            with zipfile.ZipFile(dst, "r") as zf:
                zf.extractall(path=self.targetdir)


class NATSGO():
    """Methods for formatting the gSSURGO dataset to return vectors."""

    def __init__(self, targetdir, state):
        """Initialize NATSGO object."""
        self.state = state
        self.targetdir = os.path.expanduser(os.path.abspath(targetdir))

    def __repr__(self):
        """Return representation string."""
        attrs = []
        for key, value in self.__dict__.items():
            attrs.append(f"{key}={value}")
        attrs = ", ".join(attrs)
        return f"<NATSGO object: {attrs}>"

    def build(self, variable):
        """Build a useable dataset out of NATSGO."""
        # Get the component table (may need to add tables for certain keys)
        table = self.table.copy()

        # Let's keep the horizon information
        keepers = ["lkey", "hzname", "hzdept_r", "hzdepb_r"]
        if variable not in keepers:
            table = table[keepers + [variable]]
        else:
            table = table[keepers]

        # Only keep the entries with values for our target variable
        table = table[table[variable].notna()]

        # Additionally, this may be horizon independent
        grouper = table.groupby("lkey", as_index=False)[variable]
        table["nvar"] = grouper.transform(pd.Series.nunique)
        if table["nvar"].max() == 1:
            table = table[["lkey", variable]].drop_duplicates()

        # Merge with our US shapefile
        shape = self.shape
        df = pd.merge(shape, table, on="lkey")

        # Reduce
        df = df[["geometry", variable]].drop_duplicates()

        return df

    @property
    def gdb_path(self):
        """Unpack needed elements within a state file geodatabase."""
        rfpath = f"gnatsgo/gNATSGO_{self.state}.gdb"
        return os.path.join(self.targetdir, rfpath)


    @property
    @lru_cache()
    def layers(self):
        """Return a list of available variables."""
        layers = list(fiona.listlayers(self.gdb_path))
        layers.sort()
        return layers

    @property
    @lru_cache()
    def shape(self):
        """Return the original geodatabase just for CONUS."""
        path = self.gdb_path
        shape = gpd.read_file(path, layer="SAPOLYGON")
        shape = shape.to_crs("epsg:5070")
        shape["state"] = self.state
        shape = shape.rename({"LKEY": "lkey"}, axis=1)
        return shape

    @lru_cache()
    def subtable(self, name):
        """Return the chorizon text file as a table."""
        df = gpd.read_file(self.gdb_path, layer=name)
        del df["geometry"]
        return df

    @property
    @lru_cache()
    def table(self):
        """Return the components table."""
        # In NATSGO we have overlapping map units
        laoverlap = self.subtable("laoverlap")
        muaoverlap = self.subtable("muaoverlap")

        # Get sub tables
        chorizon = self.subtable("chorizon")
        component = self.subtable("component")
        muaggatt = self.subtable("muaggatt")
        chaashto = self.subtable("chaashto")

        # Merge subtables
        table = pd.merge(chorizon, component, on="cokey")
        table = pd.merge(table, muaggatt, on="mukey")
        table = pd.merge(table, chaashto, on="chkey")

        # Now overlap with the coarser NATSGO map units
        table = pd.merge(table, muaoverlap, on="mukey", how="left")
        table = pd.merge(table, laoverlap, on="lareaovkey", how="left")

        # Stringify the keys
        table["mukey"] = table["mukey"].astype(str)
        table["lkey"] = table["lkey"].astype(str)

        return table


class SSURGO():
    """Methods for formatting the gSSURGO dataset to return vectors."""

    def __init__(self, targetdir, state):
        """Initialize SSURGO object."""
        self.state = state
        self.targetdir = os.path.expanduser(os.path.abspath(targetdir))

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
    @lru_cache()
    def chaashto(self):
        """Return the aashtocl text file as a table."""
        df = gpd.read_file(self.gdb_path, layer="chaashto")
        del df["geometry"]
        return df

    @property
    @lru_cache()
    def cogeomordesc(self):
        """Return the cogeomordesc text file as a table."""
        df = gpd.read_file(self.gdb_path, layer="cogeomordesc")
        del df["geometry"]
        return df

    @property
    @lru_cache()
    def chorizon(self):
        """Return the chorizon text file as a table."""
        df = gpd.read_file(self.gdb_path, layer="chorizon")
        del df["geometry"]
        return df

    @property
    @lru_cache()
    def component(self):
        """Return the components text file as a table."""
        df = gpd.read_file(self.gdb_path, layer="component")
        del df["geometry"]
        return df

    @property
    @lru_cache()
    def gdb_path(self):
        """Unpack needed elements within a state file geodatabase."""
        rfpath = f"gssurgo/gSSURGO_{self.state}.gdb"
        return os.path.join(self.targetdir, rfpath)

    @property
    @lru_cache()
    def layers(self):
        """Return a list of avialabe variables."""
        layers = list(fiona.listlayers(self.gdb_path))
        layers.sort()
        return layers

    @property
    @lru_cache()
    def mapunit(self):
        """Return the mapunit text file as a table."""
        df = gpd.read_file(self.gdb_path, layer="mapunit")
        del df["geometry"]
        return df

    @property
    @lru_cache()
    def muaggatt(self):
        """Return the chorizon text file as a table."""
        df = gpd.read_file(self.gdb_path, layer="muaggatt")
        del df["geometry"]
        return df

    @property
    @lru_cache()
    def shape(self):
        """Return the original geodatabase just for CONUS."""
        path = self.gdb_path
        shape = gpd.read_file(path, layer="MUPOLYGON")
        shape["state"] = self.state
        shape = shape.rename({"MUKEY": "mukey"}, axis=1)
        return shape

    @property
    @lru_cache()
    def table(self):
        """Return the components table."""
        table = pd.merge(self.chorizon, self.component, on="cokey")
        table = pd.merge(table, self.muaggatt, on="mukey")
        table = pd.merge(table, self.chaashto, on="chkey")
        table["mukey"] = table["mukey"].astype(str)
        return table



class Soil:
    """Methods for building and merging SSURGO and NATSGO layers."""

    def __init__(self, targetdir, variable):
        """Initialize Soil object."""
        self.targetdir = os.path.expanduser(os.path.abspath(targetdir))
        self.variable = variable

    def __repr__(self):
        """Return representation string."""
        attrs = []
        for key, value in self.__dict__.items():
            attrs.append(f"{key}={value}")
        attrs = ", ".join(attrs)
        return f"<SSURGO object: {attrs}>"

    def build(self, state):
        """Build a gssurgo and gnatgo layer for a variable and merge."""
        # Create SSURGO and NATSGO objects
        ssurgo = SSURGO(self.targetdir, state)
        natsgo = NATSGO(self.targetdir, state)

        # Build layers for each
        nvar = natsgo.build(self.variable)
        svar = ssurgo.build(self.variable)

        # Fill in gaps in ssurgo variable layer variable with natsgo
        var = self.merge(svar, nvar)

        return var

    def merge(self, svar, nvar):
        """Fill in gaps from the ssurgo variable layer with natsgo."""
        # Dissolve mask
        mask = svar.copy()
        mask[self.variable] = 1
        mask = mask.dissolve(self.variable)

        # Use shapely and take the difference of each national scale geometry?
        gmask = mask["geometry"].iloc[0]
        nvar["geometry"] = nvar["geometry"].parallel_apply(diff, g2=gmask)

        # Cncatenate tables?
        nvar = nvar[["geometry", self.variable]]
        svar = svar[["geometry", self.variable]]
        var = pd.concat([svar, nvar])
        del nvar
        del svar

        return var

    def main(self):
        """Build and merge natsgo and ssurgo layers for all states."""
        for state in STATES:
            lstate = state.lower()
            dst = f"../data/ssurgo/processed/{self.variable}_{lstate}.gpkg"
            natsgo = NATSGO(self.targetdir, state)
            if os.path.exists(natsgo.gdb_path):
                if not os.path.exists(dst):
                    print(f"Running ssurgo for {self.variable} in {state}.")
                    start = time.time()
                    var = self.build(state)
                    var.to_file(dst, "GPKG")
                    end = time.time()
                    minutes = round(((end - start) / 60), 2)
                    print(f"Done, {minutes} minutes.")


if __name__ == "__main__":
    variable = "brockdepmin"
    targetdir = os.path.join(HOME, "data/ssurgo")
    state = "CONUS"
    soil = Soil(targetdir, variable)
    soil.main()
