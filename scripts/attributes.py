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


TABLE_DIR = "../tables/HERE_Data_Dictionary/"
SAMPLE = "../data/state_hwys/wy_east_west.gpkg"


class Attributes:
    """Methods for accessing meta information of HERE Street attributes."""

    def __init__(self):
        """Initialize Attributes object."""
        self.sample = gpd.read_file(SAMPLE)
        self.table_dir = os.path.abspath(TABLE_DIR)

    @property
    def fields(self):
        """Return list of available fields in the HERE Streets datasets."""
        return [f.upper() for f in self.sample.columns]

    @property
    def files(self):
        """Return all html files in table dir."""
        pattern = os.path.join(self.table_dir, "**/*Table.html")
        files = glob(pattern, recursive=True)[:-1]
        return files

    @property
    @lru_cache()
    def table(self):
        """Build a master table of all attributes."""
        tables = []
        for file in self.files:
            table_list = pd.read_html(file)
            for table in table_list:
                if "Name" in table.columns:
                    tables.append(table)
        table = pd.concat(tables)
        del table["No."]
        table = table.drop_duplicates()
        table = table.sort_values("Name").reset_index(drop=True)
        return table

    @property
    def found(self):
        """Return list of found attributes."""
        table = self.table
        return [f for f in self.fields if f in table["Name"].values]

    @property
    def missing(self):
        """Return list of missing attributes."""
        table = self.table
        return [f for f in self.fields if f not in table["Name"].values]

    def main(self):
        """Extract information for all fields in the sample dataset."""


if __name__ == "__main__":
    self = Attributes()

