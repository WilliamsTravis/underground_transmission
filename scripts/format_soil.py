"""Formatting soil datasets.

Created on Sun Dec 12 11:20:03 2021

@author: travis
"""
import os

import rasterio as rio

from ssurgo import HOME


RASTER_DIR = os.path.join(HOME, "data/rasters/soils")
ORNL_DIR = os.path.join(HOME, "data/rasters/soils/global_regolith")
SSURGO_DIR = os.path.join(HOME, "data/rasters/soils/ssurgo")
LAYERS = {
    "ssurgo_brockdepmin": {
        "path": os.path.join(RASTER_DIR, "gssurgo_conus_brockdepmin.tif"),
        "orig": os.path.join(SSURGO_DIR, "gssurgo_conus_brockdepmin.tif"),
        "units": "cm"
    },
    "ssurgo_wtdepannmin": {
        "path": os.path.join(RASTER_DIR, "gssurgo_conus_wtdepannmin.tif"),
        "orig": os.path.join(SSURGO_DIR, "gssurgo_conus_wtdepannmin.tif"),
        "units": "cm"
    },
    "ornl_upland_hillslope_regolith": {
        "path": os.path.join(RASTER_DIR, "ornl_upland_hillslope_regolith.tif"),
        "orig": os.path.join(ORNL_DIR, "upland_hill-slope_regolith_thickness.tif"),
        "units": "meters"
    },
    "ornl_upland_hillslope_soil": {
        "path": os.path.join(RASTER_DIR, "ornl_upland_hillslope_soil.tif"),
        "orig": os.path.join(ORNL_DIR, "upland_hill-slope_soil_thickness.tif"),
        "units": "meters"
    },
    "ornl_valley_hillslope_fraction": {
        "path": os.path.join(RASTER_DIR, "ornl_valley_hillslope_sediment.tif"),
        "orig": os.path.join(ORNL_DIR, "hill-slope_valley-bottom.tif"),
        "units": "ratio"
    },
    "ornl_upland_valley_lowland_sediment": {
        "path": os.path.join(RASTER_DIR, "ornl_valley_hillslope_sediment.tif"),
        "orig": os.path.join(ORNL_DIR, "hill-slope_valley-bottom"),
        "units": "meters"
    },
    "ornl_average_soil_sediment": {
        "path": os.path.join(RASTER_DIR, "ornl_average_soil_sediment.tif"),
        "orig": os.path.join(ORNL_DIR, "average_soil_and_sedimentary-deposit_thickness.tif"),
        "units": "meters"
    }
}
EXTENT = ['-2356125.0', '270045.0', '2263815.0', '3172635.0']
CRS = "epsg:5070"



def formatit(layer, specs):
    """Clip, reproject, and save layer back to disk."""
    # Pull out paths
    src = specs["orig"]
    dst = specs["path"]

    # Build creation/ wo options
    options = ["tiled=yes", "compress=lzw", "blockxsize=256", "blockysize=256",
               "gdal_num_threads=11"]
    cos = ["-co " + co for co in options]
    wops = '-wo "num_threads=all_cpus"'

    # If it's in meters, convert to cm
    if specs["units"] == "meters":
        tmp = src.replace(".tif", "_temp.tif")
        cmd = f"gdal_calc.py -A {src} --calc='A*(A*100)' --outfile={tmp}"
        os.system(cmd)
        src = tmp
    else:
        tmp = None

    # Run the warp command
    cmd = " ".join(["gdalwarp", "-multi", *cos, "-te", *EXTENT, "-t_srs", CRS, src,
                    dst])
    os.system(cmd)

    # Del the temp file if meters were converted
    if tmp:
        os.remove(tmp)


def main():
    """Reformat all files."""
    for layer, specs in LAYERS.items():
        if not os.path.exists(specs["path"]):
            # break
            print(f"Formatting {layer}...")
            formatit(layer, specs)


if __name__ == "__main__":
    main()

