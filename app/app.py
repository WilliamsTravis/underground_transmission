"""DASH Portal for segmenting and characterizing right-of-ways.

Created on Wed Oct  6 08:53:08 2021

@author: travis
"""
import dash

from review.support import STYLESHEET, Logger
from flask_caching import Cache


logger = Logger()

app = dash.Dash(__name__,
                suppress_callback_exceptions=True,
                external_stylesheets=[STYLESHEET])


server = app.server

cache = Cache(
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": "data/cache",
        "CACHE_THRESHOLD": 20
    }
)

cache.init_app(server)
