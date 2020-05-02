import numpy as np
import os
import functools
import xarray as xr
import preprocess as pp
import datetime
import multiprocessing
import h5py
import glob
import tqdm

NCPU = 4
DTYPE = np.float32
BANDS = ['R', 'G', 'B', 'NIR', 'SWIR',
         'landcover', 'elevation', 'levee']
LABELBANDAXIS = 7
X_ATTR = {"desctiption": "feature array","bands": BANDS}
Y_ATTR = {"description": "contains levee in an image (1) or not (0)"}
LC_CLASSES = [[11, 12],  # water/ice
              [21, 22, 23, 24],  # developed
              [31],  # barren land
              [41, 42, 43],  # forest
              [51, 52],  # scrub
              [71, 72, 73, 74],  # grass
              [81, 82],  # crops
              [90, 95]]  # wetlands


def make_dataset(outpath, srcdir, labeltype, per_band=True):
    """
    make dataset for a model.

    Args:
        outpath (str): output data path for long-term saving.
        srcdir (str): directory to load images.
        per_band (bool): if True, process images by band separatedly.
    
    Returns:
        None
    
    Notes:
        Use per_band=True when your entire dataset cannot fit to your memory.
        This will first cache data per band, and re-read them all via
        memory-mapping which reduces memory consumption.
    """
    if per_band:
        outdir = os.path.dirname(outpath)
        outpaths, outpath_label = get_data_by_bands(srcdir, outdir,
                                                    labeltype=labeltype)
        darrays = []
        for idx, path in enumerate(outpaths):
            # re-read cached netCDF. This is lazy-loaded.
            darray = xr.open_dataset(path)[BANDS[idx]]
            darrays.append(darray)
        X = remap_bands_all(darrays)
        Y = xr.open_dataset(outpath_label)["labels"]
    else:
        X, Y = batch_process(srcdir)
        # save to netCDF for long-term storation
    save_to_netCDF(outpath, X, Y, X_ATTR, Y_ATTR)
    describe_dataset(X, Y)


def remap_bands_all(darrays):
    """
    re-map bands into one array.

    Args:
        darrays (list): list of xarray.DataArray
    
    Returns:
        xarray.DataArray
    """
    nsamples = darrays[0].shape[0]
    features_all = []
    for idx in range(nsamples):
        features = remap_bands(darrays, idx)
        features_all.append(features)
    X = xr.concat(features_all, dim="sample")
    return X


def remap_bands(darrays, idx):
    """
    re-map bands into one array.

    Args:
        darrays (list): list of xarray.DataArray
        idx (int): sample coordinate
    Returns:
        xarray.DataArray
    """
    bands = []
    for darray in darrays:
        band_data = darray.sel(sample=idx)
        bands.append(band_data)
    features = xr.concat(bands, dim="feature")
    return features


def get_data_by_bands(srcdir, outdir, labeltype="scalar"):
    """
    generate feature/label data by bands to reduce memory consumption.
    for later lazy-loading process, the output will be stored into
    netCDF4 file format instead of HDF5.

    Args:
        srcdir (str): src directory to load images
        outDir (str): output directory to store netCDF 
    """
    t = datetime.datetime.now()
    outname_band = os.path.join(outdir, 
                                "band-{0}.nc")
    outname_label = os.path.join(outdir,
                                 "labels.nc")
    outpaths = []
    for bandaxis in [0, 1, 2, 3, 4, 5, 6]:
        outpath = outname_band.format(BANDS[bandaxis])
        data = batch_process_per_band(srcdir, bandaxis)
        nsamples = data.shape[0]
        nfeatures = data.shape[1]
        nverticals = data.shape[2]
        nhorizontals = data.shape[3]
        # this DataArray is loaded on the memory, thus we cache it first.
        darray = xr.DataArray(data,
                              dims=["sample", "feature", "v", "h"],
                              coords=[np.arange(nsamples),
                                      np.arange(nfeatures),
                                      np.arange(nverticals),
                                      np.arange(nhorizontals)])
        darray.attrs["creationDate"] = t.strftime("%Y%m%d-%H:%M")
        darray.attrs["min"] = data.min()
        darray.attrs["mean"] = data.mean()
        darray.attrs["max"] = data.max()
        darray.name = BANDS[bandaxis]
        darray.to_dataset().to_netcdf(outpath)
        outpaths.append(outname_band.format(BANDS[bandaxis]))
    del darray

    Y = batch_process_per_band(srcdir, LABELBANDAXIS, labeltype=labeltype)
    if labeltype == "scalar":
        labels = xr.DataArray(Y, dims=["sample"], coords=[np.arange(nsamples)])
    elif labeltype == "array2D":
        labels = xr.DataArra(Y, dims=["sample", "v", "h"], coords=[np.arange(nsamples),
                                                                   np.arange(nverticals),
                                                                   np.arange(nhorizontals)])
    else:
        raise KeyError("undefined labeltype argument: {0}".format(labeltype))
    labels.attrs["creationDate"] = t.strftime("%Y%m%d-%H:%M")
    for key, item in Y_ATTR.items():
        labels.attrs[key] = item
    labels.name = "labels"
    labels.to_dataset().to_netcdf(outname_label)
    
    return outpaths, outname_label


def batch_process(srcDir, parallel=False, labeltype="scalar"):
    """
    batch the image processing in either serial or parallel.

    Args:
        srcDir (str): image source directory
        parallel (bool)
    
    Returns:
        np.ndarray: features
        np.ndarray: labels
    
    Notes:
        when your data is large, this (and its child methods) will take
        lots of memory.
    """
    files = glob.glob(srcDir+"/*")
    if parallel:
        files_part = [list(array) 
                      for array in np.array_split(files, NCPU)]
        with multiprocessing.Pool(NCPU) as p:
            outlist = p.map(process_images, files_part)
        outitems = parse_pooled_list(outlist)
        X = np.concatenate(outitems[0], axis=0)
        Y = np.concatenate(outitems[1], axis=0)
    else:
        outitems = process_images(files, verbose=True, labeltype=labeltype)
        X = outitems[0]
        Y = outitems[1]
    print("Feature matrix shape [samples, features, d0, d1]: ", X.shape)
    print("Label shape [samples]", Y.shape)
    # featureWiseStandardization for R, G, B, NIR, SWIR
    for i in range(0, 5):
        X[:, i, :, :] = pp.featureWiseStandardization(X[:, i, :, :])
    return X, Y


def batch_process_per_band(srcDir, bandaxis, parallel=False, labeltype="scalar"):
    """
    batch the image processing in either serial or parallel.

    Args:
        srcDir (str): image source directory
        parallel (bool)
    
    Returns:
        np.ndarray: features
        np.ndarray: labels
    
    Notes:
        when your data is large, this (and its child methods) will take
        lots of memory.
    """
    files = glob.glob(srcDir+"/*")
    if parallel:
        files_part = [list(array) 
                      for array in np.array_split(files, NCPU)]
        with multiprocessing.Pool(NCPU) as p:
            func = functools.partial(process_images_per_band, bandaxis)
            outlist = p.map(func,
                            files_part)
        data = np.concatenate(outlist, axis=0)
    else:
        data = process_images_per_band(bandaxis, files,
                                       verbose=True, labeltype=labeltype)
    # featureWiseStandardization for R, G, B, NIR, SWIR
    if bandaxis in [0, 1, 2, 3, 4]:
        # all arrays are shape[nsamples, 1, nv, nh]
        data[:, 0, :, :] = pp.featureWiseStandardization(data[:, 0, :, :])
    return data


def process_images(tiffpaths, verbose=False):
    """
    wrapper for process_image() for iteration.

    Args:
        tiffpaths (list): list of string path
    
    Returns:
        list
    """
    Xlist = []
    Ylist = []
    if verbose:
        tiffpaths = tqdm.tqdm(tiffpaths)
    for tiffpath in tiffpaths:
        X, y = process_image(tiffpath)
        Xlist.append(np.expand_dims(X,axis=0))
        Ylist.append(y)
    X_all = np.vstack(Xlist).astype(DTYPE)  # list of arrays
    Y_all = np.array(Ylist).astype(DTYPE)  # list of scalars
    return [X_all, Y_all]


def process_images_per_band(bandaxis, tiffpaths, labeltype="scalar", verbose=False):
    """
    wrapper for process_image() for iteration.

    Args:
        tiffpaths (list): list of string path
        bandaxis (int): axis to process
    
    Returns:
        numpy.ndarray
    """
    dlist = []
    if verbose:
        tiffpaths = tqdm.tqdm(tiffpaths)
    if bandaxis in [0, 1, 2, 3, 4, 5, 6]:
        for tiffpath in tiffpaths:
            data = process_image_per_band(bandaxis, tiffpath, labeltype=labeltype)
            dlist.append(np.expand_dims(data, axis=0))
        d_all = np.concatenate(dlist, axis=0).astype(DTYPE)  # list of arrays
    else:
        if labeltype == "scalar":
            for tiffpath in tiffpaths:
                data = process_image_per_band(bandaxis, tiffpath, labeltype=labeltype)
                dlist.append(data)
            d_all = np.array(dlist).astype(DTYPE)
        elif labeltype == "array2D":
            for tiffpath in tiffpaths:
                data = process_image_per_band(bandaxis, tiffpath, labeltype=labeltype)
                dlist.append(np.expand_dim(data, axis=0))
            d_all = np.concatenate(dlist, axis=0).astype(DTYPE)
    return d_all


def process_image(tiffpath, labeltype="scalar")):
    """
    pre-process image to make train/test dataset for models.

    Args:
        tiffpath (str): path to TIFF file
    
    Returns:
        numpy.ndarray: feature array X
        int: label y

    Notes:
        bands: ['R', 'G', 'B', 'NIR', 'SWIR', 
                'landcover', 'elevation', 'levee']
    """
    farray = pp.load_tiff(tiffpath)
    
    # landcover
    encoded_lc = pp.one_hot_encoding(farray[5], LC_CLASSES)
    encoded_lc = pp.remove_empty(encoded_lc)
    # print("land-cover [encoded] shape: ", encoded_lc.shape)
    # elevation
    elv = np.expand_dims(pp.sampleWiseStandardization(farray[6]), axis=0)
    # for R, G, B, NIR, SWIR 
    # perform featureWiseStandardization later
    sentinel = farray[0:5]
    X = np.vstack([sentinel, encoded_lc, elv])
    # label
    y = get_label(farray[7], labeltype=labeltype)
    return X, y


def process_image_per_band(bandaxis, tiffpath, labeltype="scalar"):
    """
    pre-process image to make train/test dataset for models.
    only process one band. use this if your entire data does not fit
    to your memory.

    Args:
        tiffpath (str): path to TIFF file
        bandaxis (int): axis to process

    Returns:
        numpy.ndarray()
    """
    farray = pp.load_tiff(tiffpath)
    if bandaxis in [0, 1, 2, 3, 4]:
        # R, G, B, NIR, SWIR
        data = np.expand_dims(farray[bandaxis], axis=0)
    elif bandaxis == 5:
        # land cover
        data = pp.one_hot_encoding(farray[5], LC_CLASSES)
        data = pp.remove_empty(data)
    elif bandaxis == 6:
        data = np.expand_dims(pp.sampleWiseStandardization(farray[6]), axis=0)
    elif bandaxis == 7:
        data = get_label(farray[7], labeltype=labeltype)
    else:
        raise KeyError("Undefined band axis {0}".format(bandaxis))
    return data


def get_label(leveeArray, labeltype="scalar", threshold=10):
    """
    count number of levee pixels and return label for a image.

    Args:
        leveeArray (numpy.ndarray): levee location array
            (1 for levee, 0/nan for non-levee)
        type (str): scalar or array2D. output label type.
        threshold (int): relevent only when type="scalar".
            minimum number of levee pixels to label as levee image
    """
    leveeArray[np.isnan(leveeArray)] = 0
    if type == "scalar":
        leveeCount = np.sum(leveeArray)
        if leveeCount > threshold:
            return 1
        else:
            return 0
    elif type == "array2D":
        return leveeArray
    else:
        raise KeyError(
            "undefined labeltype argument: {0}".format(labeltype)
            )
        

def parse_pooled_list(plist):
    """
    parse output list (of list) from multiprocessing.Pool()

    Args:
        list (list): output from Pool()
    
    Returns:
        list
    """
    inner_list_length = len(plist[0])
    items = []
    for i in range(inner_list_length):
        elems = [l[i] for l in plist]
        items.append(elems)
    return items


def save_to_netCDF(outpath, X, Y, X_attr, Y_attr):
    t = datetime.datetime.now()
    
    dshape = X.shape
    nsamples = dshape[0]
    nfeatures = dshape[1]
    nverticals = dshape[2]
    nhorizontals = dshape[3]
    features = xr.DataArray(X,
                            dims=["sample", "feature", "v", "h"],
                            coords=[np.arange(nsamples),
                                    np.arange(nfeatures),
                                    np.arange(nverticals),
                                    np.arange(nhorizontals)])
    features.attrs["creationDate"] = t.strftime("%Y%m%d-%H:%M")
    for key, item in X_attr.items():
        features.attrs[key] = item
    
    labels = xr.DataArray(Y, dims=["sample"], coords=[np.arange(Y.shape[0])])
    labels.attrs["creationDate"] = t.strftime("%Y%m%d-%H:%M")
    for key, item in Y_attr.items():
        labels.attrs[key] = item
    dset = xr.Dataset({"features": features, "labels": labels})
    dset.to_netcdf(outpath)


def describe_dataset(X, Y):
    """
    print out summary of dataset.

    Args:
        X (numpy.ndarray): feature array
        Y (numpy.ndarray): label array
    
    Returns:
        None
    """
    uniques, counts = np.unique(Y, return_counts=True)
    print("number of labels:")
    for idx, u in enumerate(uniques):
        print("\t{0}:{1}/{2}".format(u, counts[idx], Y.shape[0]))


if __name__ == "__main__":
    srcdir = "../Dataset/images/"
    outpath = "../Dataset/data.nc"
    make_dataset(outpath, srcdir, per_band=True, labeltype="array2D")