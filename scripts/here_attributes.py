"""Extract attribute information.

Created on Fri Oct  8 10:45:10 2021

@author: travis
"""
import os

from functools import lru_cache
from glob import glob

import geopandas as gpd
import pandas as pd

from tqdm import tqdm
from revruns import rr


TABLE_DIR = "../tables/HERE_Data_Dictionary/"
SAMPLE = "../data/state_hwys/wy_east_west.gpkg"
SHEET = "../tables/HERE_Data_Dictionary/HEREDataDictionaryNA_Only/data_profiles/FGDB_Plus_Data_Profiles_2020_Q3_NA_Only.xlsx"


class Attributes:
    """Methods for accessing meta information of HERE Street attributes."""

    def __init__(self):
        """Initialize Attributes object."""
        self.sample = gpd.read_file(SAMPLE)
        self.table_dir = os.path.abspath(TABLE_DIR)

    def __repr__(self):
        """Return representation string."""
        attrs = []
        for key, value in self.__dict__.items():
            attrs.append(f"{key}={value}")
        attrs = ", ".join(attrs)
        return f"<Attributes object: {attrs}>"

    @property
    def labels(self):
        """Return List of fields with their labels."""
        

    @property
    def fields(self):
        """Return list of available fields in the HERE Streets datasets."""
        return [f.upper() for f in self.sample.columns]

    @property
    def found(self):
        """Return list of found attributes."""
        table = self.table
        return [f for f in self.fields if f in table["Name"].values]

    @property
    def html_files(self):
        """Return all html files in table dir."""
        pattern = os.path.join(self.table_dir, "**/*Table.html")
        files = glob(pattern, recursive=True)[:-1]
        files = [f for f in files if "SouthAmerica" not in f]
        return files

    @property
    def missing(self):
        """Return list of missing attributes."""
        table = self.table
        return [f for f in self.fields if f not in table["Name"].values]

    @property
    @lru_cache()
    def table(self):
        """Build a master table of all attributes."""
        tables = []
        for file in self.html_files:
            table_list = pd.read_html(file)
            for table in table_list:
                if "Name" in table.columns:
                    tables.append(table)

        table = pd.concat(tables)
        table = table[["Name", "Full Name", "Description"]]
        table = table.drop_duplicates()
        table = table.sort_values("Name").reset_index(drop=True)

        return table

    def main(self):
        """Extract information for all fields in the sample dataset."""


if __name__ == "__main__":
    self = Attributes()
