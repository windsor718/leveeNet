# leveeNet: Deep learning based manmade-levee detection project  
Preliminary version   
## Project description  
trying to detect two things:  
- **where** does levees locate?  
- **what** are the properties (e.g., height) of those levees?
As of 2020/04, currently am working in the first bullet, **where** are levees.  
For the very first step of this project, the current goal is to predict whether the image is protected-river or not.  
## Change log  
2020/04/26 created repository.  
## Source codes  
- **gee**: scripts for processing in/out-of Google Earth Engine  
- **preprocess**: scripts for preprocessing output images from gee  
## Workflow  
### Google Earth Engine processing  
To generate images to train a model, process satellite-images on Google Earth Engine.  
The dataset currently used:  
- [Sentinel-2 MSI: MultiSpectral Instrument, Level-2A](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR)
- [NLCD: USGS National Land Cover Database](https://developers.google.com/earth-engine/datasets/catalog/USGS_NLCD)
- [USGS National Elevation Dataset 1/3 arc-second](https://developers.google.com/earth-engine/datasets/catalog/USGS_NED)
- [National Levee Database](https://levees.sec.usace.army.mil/#/) by USACE  
- [WWF HydroSHEDS Free Flowing Rivers Network v1](https://developers.google.com/earth-engine/datasets/catalog/WWF_HydroSHEDS_v1_FreeFlowingRivers)  
  
The images are generated by following steps:  
1. locate levees (from National Levee Database) on the map.  
2. locate hydrography (rivers) on the map.  
3. from single river centerline at a cross-section (line perpendicular to the centerline), draw a buffered-circle and its bounded rectangle. This bounded rectangle is the bbox of output image.  
4. map all other dataset and aggregate them into one image with multuiple bands.  
5. clip the image with bboxes, and output as a GeoTIFF file.  

As this process analyzes millions of satellite-images, the downloading process takes long time. We can potentially have thousands of images by this process, but for this reason as of 2020/04 I limited the number of processed regions by basin and stream order.  
  
### Image preprocessing  
After downloading GeoTIFF files, preprocess data to perform standardization and create labels for Keras-ready format.  
While Keras has some utilities for preprocessing, as we have multiple bands different from normal images we need to do this by outselves.  
1. standardization 
   - elevation: *sampleWiseStandardization*. We are more interested in the variation of the elevation in a image, not an entire dataset. 
     - For instance, there should be differences in mean elevation between images from mountain and near-ocean, but this difference of mean may not be relavent.   
   - bands from Sentinel-2: *featureWiseStandardization*. this is more like global variable, thus perform standardiztion for entire dataset.  
2. one-hot-encoding
   - as the land cover band (channel) is categorical, create dummy variables via one-hot-encoding. 
   - We also need to remove homogeneous layers (i.e., all-zero).  
3. labeling: make labels from the levee layer.  
4. output to HDF5.  
