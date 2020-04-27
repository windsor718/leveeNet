"""
Javascript version of this code for interactive visualization:
https://code.earthengine.google.com/f5137a302d2c649021ca9e78461c9467
As this procedure is highly context dependent,
the coding is mostly done in the global scope.
Some styles are inconsistent with PEP8 standards as this script follows
GEE-JS standard.
"""
import ee
import os
import functools
import configparser
import time
from geeutils import geeutils as gu


configpath = "./config.ini"

def load_config(configpath):
    parser = configparser.ConfigParser()
    parser.read(configpath)
    description = parser.get("config", "description")
    radius = int(parser.get("config", "radius"))
    dimensions = parser.get("config", "dimensions")
    folder = parser.get("config", "folder")
    paramdict = {"description": description,
                 "radius": radius,
                 "dimensions": dimensions,
                 "folder": folder}
    return paramdict


def maskS2clouds(image):
  qa = image.select('QA60')
  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloudBitMask = 1 << 10
  cirrusBitMask = 1 << 11
  #Both flags should be set to zero, indicating clear conditions.
  mask = (qa.bitwiseAnd(cloudBitMask).eq(0)
            .And(qa.bitwiseAnd(cirrusBitMask).eq(0)))
  return image.updateMask(mask).divide(10000)


def generate_square_by_scale(scale, feature):
    centroid = feature.geometry().centroid()
    square = centroid.buffer(scale).bounds()
    return ee.Feature(square)


def generate_imageList(image, feature):
    outimage = (image.clip(feature.geometry())
                     .setMulti({"id": "leveeDetectionDataset"}))
    return outimage


# --main--
params = load_config(configpath)
# ---server-side [define-and-run]---
# load datasets
utils = gu.utils()
utils.start_session()
sentinel = (ee.ImageCollection('COPERNICUS/S2_SR')
              .filterDate('2019-01-01', '2020-03-30')
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
levee = ee.FeatureCollection("users/winzer718/MSR_levee_line")
landcover = ee.ImageCollection('USGS/NLCD')
elevation = ee.Image('USGS/NED')
basin = ee.FeatureCollection("users/winzer718/MSR_basin_boundary")
sheds = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers")

# aggregate ImageCollection to get Image
sentinelImg = (sentinel.map(maskS2clouds)
                       .median().select(["B4", "B3", "B2", "B8", "B11"],
                                        ["R", "G", "B", "NIR", "SWIR"]))
leveeImg = levee.reduceToImage(
                  properties=["id"],
                  reducer=ee.Reducer.first()
                ).rename("levee")
lcImg = (landcover.select('landcover')
                  .mosaic())
### for land cover, images are separated with different regions. Thus .mosaic() here.
elvImg = elevation  # already image, just for terminology

# extract desired regions along with rivers
# hydro = (sheds.filterBounds(basin)
#               .filter(ee.Filter.lt("RIV_ORD", 4)))
hydro = (sheds.filterBounds(basin)
              .filter(ee.Filter.gt("RIV_ORD", 3)).filter(ee.Filter.lt("RIV_ORD", 5)))
generate_square = functools.partial(generate_square_by_scale, params["radius"])
outsquares = hydro.map(generate_square)

# merge them into one image with multiple bands--
outImage = (sentinelImg.addBands(lcImg)
                       .addBands(elvImg)
                       .addBands(leveeImg))

# list of outputs for desired regions
generator = functools.partial(generate_imageList, outImage)
images = outsquares.map(generator)

# ---client-side [difine-by-run]---
tasks = utils.define_tasks(images,
                           params["description"],
                           params["dimensions"],
                           params["folder"])
[task.start() for task in tasks]
while True:
    print(tasks[-1].status())
    time.sleep(30)
