"""ROW navigation bar.

Created on Wed Oct  6 09:12:53 2021

@author: travis
"""
import dash_core_components as dcc
import dash_html_components as html

from review.support import BUTTON_STYLES


NAVBAR = html.Nav(
    className="top-bar fixed",
    children=[

        html.Div([

          html.Div([
              html.H1(
                  "ROW Viewer | ",
                  style={
                     'float': 'left',
                     'position': 'relative',
                     "color": "white",
                     'font-family': 'Times New Roman',
                     'font-size': '48px',
                     "font-face": "bold",
                     "margin-bottom": 5,
                     "margin-left": 15,
                     "margin-top": 0
                     }
              ),
              html.H2(
                  children=("  Segmented Right-of-Way Characterization"),
                  style={
                    "float": "left",
                    "position": "relative",
                    "color": "white",
                    "font-family": "Times New Roman",
                    "font-size": "28px",
                    "margin-bottom": 5,
                    "margin-left": 15,
                    "margin-top": 15,
                    "margin-right": 55
                  }
                ),
              ]),

          html.A(
            html.Img(
              src=("/static/nrel_logo.png"),
              className="twelve columns",
              style={
                  "height": 70,
                  "width": 180,
                  "float": "right",
                  "position": "relative",
                  "margin-left": "10",
                  "border-bottom-right-radius": "3px"
                  }
            ),
            href="https://www.nrel.gov/",
            target="_blank"
          )

          ],
            style={
                "background-color": "#1663B5",
                "width": "100%",
                "height": 70,
                "margin-right": "0px",
                "margin-top": "-15px",
                "margin-bottom": "15px",
                "border": "3px solid #FCCD34",
                "border-radius": "5px"
                },
            className="row"),
])
