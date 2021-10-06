"""Layout for right of way viewer.

Created on Wed Oct  6 09:09:14 2021

@author: travis
"""
import copy
import json

import dash_core_components as dcc
import dash_html_components as html

from review.support import (BASEMAPS, BOTTOM_DIV_STYLE, BUTTON_STYLES,
                            CHART_OPTIONS, COLOR_OPTIONS, DEFAULT_MAPVIEW,
                            REGION_OPTIONS, STATES, TAB_STYLE, TABLET_STYLE)


# Everything below goes into a css
TABLET_STYLE_CLOSED = {
    **TABLET_STYLE,
    **{"border-bottom": "1px solid #d6d6d6"}
}
TAB_BOTTOM_SELECTED_STYLE = {
    'borderBottom': '1px solid #1975FA',
    'borderTop': '1px solid #d6d6d6',
    'line-height': '25px',
    'padding': '0px'
}

RC_STYLES = copy.deepcopy(BUTTON_STYLES)
RC_STYLES["off"]["border-color"] = RC_STYLES["on"]["border-color"] = "#1663b5"
RC_STYLES["off"]["border-width"] = RC_STYLES["on"]["border-width"] = "3px"
RC_STYLES["off"]["border-top-width"] = "0px"
RC_STYLES["on"]["border-top-width"] = "0px"
RC_STYLES["off"]["border-radius-top-left"] = "0px"
RC_STYLES["on"]["border-radius-top-right"] = "0px"
RC_STYLES["off"]["float"] = RC_STYLES["on"]["float"] = "right"
# Everything above goes into css


LAYOUT = html.Div(
    children=[

        # Summary statistics
        html.Div(id="summary", children="Summary Stats: "),

        # A button for later
        html.Button(id="submit", children="Button"),
        html.Hr(style={"margin-bottom": "50px"}),

        # Map and Chart
        html.Div(
            className="row",
            children=[
                dcc.Graph(id="map", className="six columns"),
                dcc.Graph(id="chart", className="six columns")
            ]
        ),

        # Storage
        html.Div(id="mapview", style={"display": "false"})

    ]
)
