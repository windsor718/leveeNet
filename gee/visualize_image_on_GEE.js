/*
Extract features from Google Earth Engine
This script is for interectively visualizing the
features. To batch output images, use python API.
*/

// -output options-
var radius = 5000 // meters
var dimensions = "1000x1000" // output dimension WxH

// -functions and utilities-
function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

function generate_square_by_scale(scale){
  return function generate_square_along_river(feature) {
    var centroid = feature.geometry().centroid();
    var square = centroid.buffer(scale).bounds();
    return ee.Feature(square);
  }
}

function generate_imageList(image){
  return function(feature){
    var outimage = image.clip(feature.geometry())
                        .setMulti({
                          id: "leveeDetectionDataset"
                        })
    return outimage
  }
}

var rgbVis = {
  min: 0.0,
  max: 0.3,
  bands: ['R', 'G', 'B'],
};

var landcoverVis = {
  min: 0.0,
  max: 95.0,
  palette: [
    '000000', '000000', '000000', '000000',
    '000000', '000000', '000000', '000000',
    '000000', '000000', '000000', '466b9f',
    'd1def8', '000000', '000000', '000000',
    '000000', '000000', '000000', '000000',
    '000000', 'dec5c5', 'd99282', 'eb0000',
    'ab0000', '000000', '000000', '000000',
    '000000', '000000', '000000', 'b3ac9f',
    '000000', '000000', '000000', '000000',
    '000000', '000000', '000000', '000000',
    '000000', '68ab5f', '1c5f2c', 'b5c58f',
    '000000', '000000', '000000', '000000',
    '000000', '000000', '000000', 'af963c',
    'ccb879', '000000', '000000', '000000',
    '000000', '000000', '000000', '000000',
    '000000', '000000', '000000', '000000',
    '000000', '000000', '000000', '000000',
    '000000', '000000', '000000', 'dfdfc2',
    'd1d182', 'a3cc51', '82ba9e', '000000',
    '000000', '000000', '000000', '000000',
    '000000', 'dcd939', 'ab6c28', '000000',
    '000000', '000000', '000000', '000000',
    '000000', '000000', 'b8d9eb', '000000',
    '000000', '000000', '000000', '6c9fb8'
  ],
};

var elevationVis = {
  min: 0.0,
  max: 4000.0,
  gamma: 1.6,
};


// -main processing-

// --load datasets to encode--
var sentinel = ee.ImageCollection('COPERNICUS/S2_SR')
                 .filterDate('2019-01-01', '2020-03-30')
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));
var levee = ee.FeatureCollection("users/winzer718/MSR_levee_line");
var landcover = ee.ImageCollection('USGS/NLCD');
var elevation = ee.Image('USGS/NED');
var basin = ee.FeatureCollection("users/winzer718/MSR_basin_boundary")
var sheds = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers")

// --aggregate ImageCollection to get Image--
var sentinelImg = sentinel.map(maskS2clouds)
                          .median()
                          .select(["B4", "B3", "B2", "B8", "B11"],
                                  ["R", "G", "B", "NIR", "SWIR"])
var leveeImg = levee.reduceToImage({
                  properties: ["id"],
                  reducer: ee.Reducer.first()
                }).rename("levee");
var lcImg = landcover.select('landcover')
                     .mosaic();
// for land cover, images are separated with different regions. Thus .mosaic() here.
var elvImg = elevation  // already image, just for terminology

// --extract desired regions along with rivers--
var hydro = sheds.filterBounds(basin)
                 .filter(ee.Filter.lt("RIV_ORD", 4))
var generate_square = generate_square_by_scale(radius)
var outsquares = hydro.map(generate_square)

// --visualization--
Map.addLayer(sentinelImg, rgbVis, "Sentinel-2")
Map.addLayer(lcImg, landcoverVis, "land cover")
Map.addLayer(elvImg, elevationVis, "elevation")
Map.addLayer(leveeImg, {}, "levee")
Map.addLayer(outsquares, {}, "square")

// --merge them into one image with multiple bands--
var outImage = sentinelImg.addBands(lcImg)
                          .addBands(elvImg)
                          .addBands(leveeImg)
// print(outImage)

// --list of outputs for desired regions--
var output = generate_imageList(outImage)
var images = outsquares.map(output)

// --client side operation--
// It is too heavy to batch the operation via JS
// Use Python API to batch them.
// var n = images.size().getInfo();
// var imgs = images.toList(n)
// var im = ee.Image(imgs.get(0));
// var prefix = "0_js";
// var description = "leveeDetectionDataset";
// Map.addLayer(im);
// Export.image.toDrive({
//   image: im.float(),
//   description: description,
//   dimensions: "1000x1000",
//   region: im.geometry(),
//   fileNamePrefix: prefix,
//   fileFormat: "GeoTIFF"
// });