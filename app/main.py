"""ROW callbacks.

Created on Wed Oct  6 09:20:24 2021

@author: travis
"""
import copy
import json
import os

from functools import lru_cache

import dash
import dash_core_components as dcc
import dash_html_components as html
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px

from app import app, cache, logger
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from layout import LAYOUT, RC_STYLES
from navbar import NAVBAR
from review.support import (AGGREGATIONS, BUTTON_STYLES, COLOR_OPTIONS,
                            COLOR_Q_OPTIONS, COLORS, COLORS_Q,
                            MAP_LAYOUT, TABLET_STYLE)

DEFAULT_MAPVIEW = {
    "mapbox.center": {
        "lon": -104.87,
        "lat": 39.25
    },
    "mapbox.zoom": 4.8,
    "mapbox.bearing": 0,
    "mapbox.pitch": 0
}
EW = gpd.read_file("../data/state_hwys/co_east_west.gpkg")
NS = gpd.read_file("../data/state_hwys/co_north_south.gpkg")
# EW = gpd.read_file("../data/ew_interstates.gpkg")
# NS = gpd.read_file("../data/ns_interstates.gpkg")


def point_filter(df, selection):
    """Filter a dataframe by points selected from the chart."""
    if selection:
        points = selection["points"]
        gids = [p["customdata"][0] for p in points]
        df = df[df["gid"].isin(gids)]
    return df, gids


class Maps:
    """Methods for building plotly map plots."""

    def __init__(self, mapview=DEFAULT_MAPVIEW, mapsel=None, ns=NS, ew=EW):
        """Initialize Maps."""
        self.mapsel = mapsel
        self.ns = ns
        self.ew = ew

    # @lru_cache(1)
    def format_df(self, df):
        """Format the geodataframes into plottable tables."""
        rows = df.apply(self.split_geometries, axis=1)  # parallelize?
        rows = [r for sr in rows for r in sr]
        df = pd.DataFrame(rows)
        df["hover"] = df.apply(lambda x: f"{x['latitude']}, {x['longitude']}",
                               axis=1)
        return df

    def layout(self, mapview, basemap="light", title="MAP", ylims=[0, 100]):
        """Build the map data layout dictionary."""
        layout = copy.deepcopy(MAP_LAYOUT)
        layout["mapbox"]["center"] = mapview["mapbox.center"]
        layout["mapbox"]["zoom"] = mapview["mapbox.zoom"]
        layout["mapbox"]["bearing"] = mapview["mapbox.bearing"]
        layout["mapbox"]["pitch"] = mapview["mapbox.pitch"]
        layout["mapbox"]["style"] = basemap
        layout["title"]["text"] = title
        layout["yaxis"] = dict(range=ylims)
        layout["legend"] = dict(
            title_font_family="Times New Roman",
            bgcolor="#E4ECF6",
            font=dict(
                family="Times New Roman",
                size=15,
                color="black"
            )
        )
        return layout

    def split_geometries(self, row, buffer=30):
        """Split a line geometry into nodes. No buffer yet."""
        # Build an id for this segment
        sid = f"{row['FromNodeID']}_{row['ToNodeID']}"

        # Extract coordinate pairs
        coords = [p.coords.xy for p in row["geometry"]][0]
        xs = coords[0]
        ys = coords[1]

        # Copy row for each coordinate pair
        rows = []
        for i, x in enumerate(xs):
            nrow = row.copy()
            nrow["latitude"] = ys[i]
            nrow["longitude"] = x
            nrow["sid"] = sid
            rows.append(nrow)

        return rows

    @property
    def dataframe(self):
        """Return the full dataframe."""
        test_path = "../data/test.csv"
        if not os.path.exists(test_path):
            ns = self.format_df(self.ns)
            ns["direction"] = "North-South"
            ns["color"] = "blue"
            ew = self.format_df(self.ew)
            ew["direction"] = "East-West"
            ew["color"] = "orange"
            df = pd.concat([ns, ew])
            df.to_csv(test_path, index=False)
        else:
            df = pd.read_csv(test_path)
        df["gid"] = df.index
        if self.mapsel:
            _, gids = point_filter(df, self.mapsel)
            df["color"][df["gid"].isin(gids)] = "red"

        return df

    @property
    def distance(self):
        """Return total distance represented."""
        df = self.dataframe
        if self.mapsel:
            df, _ = point_filter(df, self.mapsel)
        meters = df["Meters"].sum()
        km = meters / 1_000
        km = int(round(km, 0))
        return km

    def build(self):
        """Build plotly map figure."""
        df = self.dataframe

        # figure = px.line_mapbox(df, lat="latitude", lon="longitude",
        #                         hover_name="StreetName", color="sid",
        #                         line_group="sid")

        figure = px.scatter_mapbox(df, lat="latitude", lon="longitude",
                                   hover_name="StreetName", color="color",
                                   custom_data=["gid"])

        figure.update_layout(**self.layout(DEFAULT_MAPVIEW))

        return figure


@cache.memoize()
def get_figure():
    """Cache retrieve and dataframe."""
    

@app.callback([Output("map", "figure"),
               Output("mapview", "children"),
               Output("summary", "children")],
              [Input("submit", "n_clicks"),
               Input("map", "selectedData")],
              [State("mapview" ,"children"),
               State("map", "relayoutData")])
def make_map(submit, mapsel, mapstore, mapview):
    """Build map."""
    print(f"Making Map: {submit}")
    logger.setargs(make_map, submit, mapsel, mapstore, mapview)

    # Maintain view on updates
    if not mapview:
        mapview = DEFAULT_MAPVIEW
    elif 'mapbox.center' not in mapview.keys():
        mapview = DEFAULT_MAPVIEW

    # Build figure
    mapper = Maps(mapview, mapsel=mapsel)
    figure = mapper.build()

    # Make summary
    summary = "Summary Stats: {:,}km".format(mapper.distance)

    return figure, json.dumps(mapview), summary


if __name__ == "__main__":
    self = Maps()
