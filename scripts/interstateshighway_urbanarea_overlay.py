# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import geopandas as gpd
import numpy as np

def highway_ua(uadir, uaname, crs, hwdir, hwname, outdir, output):
    os.chdir(uadir)
    ua = gpd.read_file(uaname)
    ua.crs
    ua = ua.to_crs(crs)
    print(ua.crs)
    # ua.columns
    # ua.UATYP10.unique()
    os.chdir(hwdir)
    hw = gpd.read_file(hwname)
    print(hw.crs)
    
    join = hw.sjoin(ua, how = 'left')
    # join.columns
    print(join.UATYP10.unique())
    join.UATYP10 = join.UATYP10.replace({np.nan: 'none'})
    
    
    join.to_file(output, driver='GPKG')
    print('output generated!')
    
uadir = '/shared-projects/rev/projects/lpo/fy21/underground_transmission/data/census/cb_2018_us_ua10_500k'
uaname = 'cb_2018_us_ua10_500k.shp'
crs = 'epsg:4326'
hwdir = '/shared-projects/rev/projects/lpo/fy21/underground_transmission/data'
ewhw = 'ew_interstates_geomcorrect_final.gpkg'
outdir = '/shared-projects/rev/projects/lpo/fy21/underground_transmission/data'
output = 'ew_interstates_geomcorrect_ua.gpkg'

highway_ua(uadir, uaname, crs, hwdir, ewhw, outdir, output)

uadir = '/shared-projects/rev/projects/lpo/fy21/underground_transmission/data/census/cb_2018_us_ua10_500k'
uaname = 'cb_2018_us_ua10_500k.shp'
crs = 'epsg:4326'
hwdir = '/shared-projects/rev/projects/lpo/fy21/underground_transmission/data'
nshw = 'ns_interstates_geomcorrect_final.gpkg'
outdir = '/shared-projects/rev/projects/lpo/fy21/underground_transmission/data'
output = 'ns_interstates_geomcorrect_ua.gpkg'

highway_ua(uadir, uaname, crs, hwdir, nshw, outdir, output)


os.chdir('/shared-projects/rev/projects/lpo/fy21/underground_transmission/data/census/cb_2018_us_ua10_500k')
ua = gpd.read_file('cb_2018_us_ua10_500k.shp')
ua.crs
ua = ua.to_crs('epsg:4326')
ua.crs
# ua.columns
# ua.UATYP10.unique()
os.chdir('/shared-projects/rev/projects/lpo/fy21/underground_transmission/data')
hw = gpd.read_file('ew_interstates_geomcorrect_final.gpkg')
hw.crs

join = hw.sjoin(ua, how = 'left')
join.columns
join.UATYP10.unique()
join.UATYP10 = join.UATYP10.replace({np.nan: 'none'})


join.to_file('ew_interstates_geomcorrect_ua.gpkg', driver='GPKG')

