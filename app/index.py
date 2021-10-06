"""ROW index page.

Created on Wed Oct  6 09:15:36 2021

@author: travis
"""
import dash_core_components as dcc
import dash_html_components as html

import main

from app import app, server, logger
from navbar import NAVBAR


app.layout = html.Div([
    # dcc.Location(id="url", refresh=False),
    NAVBAR,
    main.LAYOUT
])


if __name__ == '__main__':
    # app.run_server(debug=True, port="8050")
    app.run_server(debug=False, port="8050")
