"""
some additional processes before training U-Net.
"""
import numpy as np
import xarray as xr
import argparse
import os

np.random.seed(1)


def make_dataset(datapath, outdir, testsplit=0.8, validsplit=0.8):
    """
    make dataset for u-net training/testing.

    Args:
        datapath (str): path to the dataset.
            created by preprocess/prepare_data.py
        outdir (str): root directory to store outputs.
        testplit (float): ratio of training dataset out of the entire dataset.
        validsplit (float): ratio of validation dataset
            out of the dataset without test datast.

    Returns:
        None
    """
    data = xr.open_dataset(datapath)
    X = data["features"]
    Y = data["labels"]
    labels1D = make_labels_from_images(Y)  # reference for filtering

    # filter with labels
    X_filtered, _ = filter_with_label(X, labels1D)
    Y_filtered, _ = filter_with_label(Y, labels1D)

    # split dataset
    indices = X_filtered["sample"].values
    train_idx, valid_idx, test_idx \
        = split_dataset(indices, testsplit, validsplit)
    X_train = X_filtered.sel(sample=train_idx)
    Y_train = Y_filtered.sel(sample=train_idx)
    X_valid = X_filtered.sel(sample=valid_idx)
    Y_valid = Y_filtered.sel(sample=valid_idx)
    X_test = X_filtered.sel(sample=test_idx)
    Y_test = Y_filtered.sel(sample=test_idx)

    # output training images
    cache_as_single_images(X_train, Y_train,
                           os.path.join(outdir, "train"))
    cache_as_single_images(X_valid, Y_valid,
                           os.path.join(outdir, "valid"))
    cache_as_single_images(X_test, Y_test,
                           os.path.join(outdir, "test"))


def split_dataset(indices, testsplit, validsplit):
    """
    split indices into three.
    |-----------data-----------|
    |-----testsplit-----|------|
    |---validsplit---|--|

    Args:
        indices (numpy.array, list): indices to split
        testsplit (float): ratio of training dataset out of the entire dataset.
        validsplit (float): ratio of validation dataset
            out of the dataset without test datast.

    Returns:
        numpy.ndarray: training indices
        numpy.ndarray: validation indices
        numpy.ndarray: test indices
    """
    indices = np.array(indices)

    np.random.shuffle(indices)
    nsamples = indices.shape[0]
    tval_indices = indices[0:int(nsamples*testsplit)]
    test_indices = indices[int(nsamples*testsplit)::]

    np.random.shuffle(tval_indices)
    nsamples_t = tval_indices.shape[0]
    train_indices = tval_indices[0:int(nsamples_t*validsplit)]
    valid_indices = tval_indices[int(nsamples_t*validsplit)::]

    return train_indices, valid_indices, test_indices


def cache_as_single_images(X, Y, outdir):
    """
    split multi-sample dataset into single images.

    Args:
        X (xarray.DataArray): features
        Y (xarray.DataArray): labels

    Returns:
        None
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    samples = X["sample"].values.tolist()
    for sample in samples:
        x = X.sel(sample=sample)
        y = Y.sel(sample=sample).astype(np.int32)
        dset = xr.Dataset({"features": x, "labels": y})
        dset.to_netcdf(os.path.join(outdir, "{0:05d}.nc".format(sample)))


def filter_with_label(X_darray, Y_darray, labels=[1]):
    """
    return feature/label array only with one specific label.

    Args:
        X_darray (xarray.DataArray): input features
        Y_darray (xarray.DataArray): labels (1D; [N])
        labels (list): labels to extract

    Returns:
        xarray.DataArray: extracted data
        xarray.DataArray: extracted data
    """
    Y_selected = Y_darray.where(Y_darray.isin(labels), drop=True)
    indices = Y_selected["sample"].values
    X_selected = X_darray.sel(sample=indices)
    return X_selected, Y_selected


def make_label_from_image(Y_darray, threshold=10):
    """
    make scalar label for filtering.

    Args:
        Y_darray (xarray.DataArray): binary value image (0, 1).
            1 is the interested target .
        threshold (int): the threshold of pixels containing 1.
            if exceeds, then label as 1, else 0.
    """
    leveeCount = np.sum(Y_darray)
    if leveeCount > threshold:
        return 1
    else:
        return 0


def make_labels_from_images(Y_darray):
    samples = Y_darray["sample"].values.tolist()
    labels = []
    for sample in samples:
        Y = Y_darray.sel(sample=sample)
        label = make_label_from_image(Y)
        labels.append(label)
    labels = xr.DataArray(labels, dims=["sample"], coords=[samples])
    return labels


if __name__ == "__main__":
    datapath = "~/work/leveeNet/Dataset/segmentation_data.nc"
    outdir = "~/work/leveeNet/Dataset/images/"
    make_dataset(datapath, outdir)
