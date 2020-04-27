import numpy as np
import xarray as xr


def load_tiff(tiffpath):
    """
    load GeoTIFF image from a file.

    Args:
        tiffpath (str): path to tiff file.
    
    Returns:
        numpy.ndarray
    """
    data = xr.open_rasterio(tiffpath)
    return data.values


def rescale(array, scale):
    """
    rescale array by multiplyting a scaler.

    Args:
        array (numpy.ndarray): input array to rescale
        scale (int or float): scaler value to multiply
    
    Returns:
        numpy.ndarray
    """
    return array*scale


def one_hot_encoding(array2d, classes):
    """
    generate new array with N(classes) dimension.

    Args:
        array2d (numpy.ndarray): input 2d array with categorical values
        classes (list): list of categorical values. 
            If merge=True, list of list of categorical values,
            whose inner-list is the variables to be merged.
    
    Returns:
        numpy.ndarray (shape[N, Height, Width])
    """
    layers = []
    for c in classes:
        if isinstance(c, list):
            layer = np.where(np.isin(array2d, c), 1, 0)
        else:
            layer = np.where(array2d==c, 1, 0)
        layers.append(np.expand_dims(layer, axis=0))
    return np.vstack(layers)


def remove_empty(array, axis=0):
    """
    remove empty (all 0) layer after one-hot encoding.

    Args:
        array (numpy.ndarray): one-hot encoded array
        axis (int): axis of layers
    
    Returns:
        numpy.ndarray: the dimension in the specified axis
            is reduced to M < N.
    """
    remove_idx = []
    for l in range(array.shape[axis]):
        if (array.sum(axis=axis) == 0).all():
            remove_idx.append(l)
    return np.delete(array, remove_idx, axis=axis)


def sampleWiseStandardization(array2d):
    """
    standardize data in sample space in a band (=2d).

    Args:
        array2d (numpy.ndarray): input 2d array to standardize.
    
    Returns:
        numpy.ndarray: array with mean 0, std 1.
    """
    assert array2d.ndim == 2, "Input diemsion must be 2"
    mean = array2d.mean()
    std = array2d.std()
    return (array2d-mean)/std


def featureWiseStandardization(array3d):
    """
    standardize data in feature space in a band (=3d)

    Args:
        array3d (numpy.ndarray): input 3d ([samples, h, w]) array to standardize
    
    Returns:
        numpy.ndarray: array with mean 0, std 1.
    """
    assert array3d.ndim == 3, "Input dimension must be 3"
    mean = array3d.mean()
    std = array3d.std()
    return (array3d-mean)/std


