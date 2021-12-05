#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:31:56 2021

@author: yren
"""
import psycopg2
import geopandas as gpd
import pandas as pd
import os
import gc
# set up pg db con
host = 'gds_publish.nrel.gov'
database = 'ref_lightbox'

username = "yren"
password = 'kjw5PEe8wjZO'
## get connection to the Postgrd SQL database
con = psycopg2.connect(database = database, user=username, password=password, host=host)

sql = f"select distinct(state) from parcels.parcels ;"
test = pd.read_sql_query(sql, con)
states = test['state']

outer_states = ['guam', 'alaska', 'virgin_islands', 'hawaii', 'northern_mariana_islands', 'puerto_rico']

for state in test['state'][23:]:
    if state in outer_states:
        print(state)
    else:
        print('working on......', state)
        # select state, geom and transform to 5070, and use_code_std_ctgr_desc_lps; save as gpkg to hpc
        sql = f"SELECT state, use_code_std_ctgr_desc_lps, the_geom_4326 from parcels.parcels WHERE state='{state}'"
        
        state_parcel = gpd.GeoDataFrame.from_postgis(sql, con, geom_col='the_geom_4326')
        
        state_parcel.crs
        state_parcel = state_parcel.to_crs('epsg:5070')
        state_parcel = state_parcel.rename(columns={'the_geom_4326':'the_geom_5070'})
        state_parcel = gpd.GeoDataFrame(state_parcel, geometry = 'the_geom_5070')
        os.chdir(r'/projects/rev/data/conus/lightbox/lightbox_parcels_ownership')
        state_parcel.to_file( state +'_5070.gpkg', driver='GPKG',layer='parcels')
        print(state, '... DONE!')
        del(state_parcel)
        gc.collect()
