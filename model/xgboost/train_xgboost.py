import numpy as np
import xarray as xr
import xgboost as xgb
import time
import os
from sklearn.metrics import accuracy_score, roc_curve, auc


XGB_DEFAULT_PARAMS = {"max_depth": 6, "eta": 0.3,
                      "objective": "binary:logistic",
                      "threads": 4, "eval_metric": "error",
                      "early_stopping_rounds": 100,
                      }


def load_netcdf(ncpath):
    """
    load netcdf datasets named "features" and "labels".

    Args:
        ncpath (str): netcdf path

    Returns:
        xarray.DataArray: features
        xarray.DataArray: labels
    """
    data = xr.open_dataset(ncpath)
    X = data["features"]
    Y = data["labels"]
    return X, Y


def match_nsamples(X_darray, Y_darray):
    """
    down-sample the majority data. The target number is the
    minimum of the minority count.

    Args:
        X_darray (xarray.DataArray): input features
        Y_darrat (xarray.DataArray): labels (1D; [N])

    Returns:
        xarray.DataArray: downsampled data
        xarray.DataArray: downsampled data
    """
    Y = Y_darray.values
    uniques, counts = np.unique(Y, return_counts=True)
    minCount = counts.min()
    out_indices = []
    for u in uniques.tolist():
        org_indices = np.where(Y == u)[0]
        ds_indices = np.random.choice(org_indices, size=minCount)
        out_indices.append(ds_indices)
    sampled_indices = np.concatenate(out_indices)
    return X_darray.isel(sample=sampled_indices), Y_darray.isel(sample=sampled_indices)


def reduce_dimension(darray, operation="mean", dims=["v", "h"]):
    """
    reduce spatial matrix into a scalar.

    Args:
        darray (xarray.DataArray): input array [nsamples, v, h]
        operation (str): reduction method

    Returns:
        darray (xarray.DataArray): [nsamples]
    """
    if operation == "mean":
        return darray.mean(dims)
    elif operation == "sum":
        print(darray.sum(dims))
        return darray.sum(dims)
    elif operation == "max":
        return darray.max(dims)
    elif operation == "min":
        return darray.min(dims)
    elif operation == "std":
        return darray.std(dims)
    else:
        raise KeyError("undefined operation. {0}".format(operation))


def standardize(featurearray, format="sample_last"):
    if format == "sample_last":
        axis = 0
    elif format == "sample_first":
        axis = 1
    else:
        raise KeyError("undefined format: {0}".format(format))
    new_featurearray = \
        (featurearray - featurearray.mean(axis=axis))/featurearray.std(axis=axis)
    return new_featurearray


def make_dataset(ncpath):
    X, Y = load_netcdf(ncpath)
    X, Y = match_nsamples(X, Y)  # down sampling
    Xs = []
    for c in [0, 1, 2, 3, 4]:
        # bands from sentinel [R, G, B, NIR, SWIR]
        x = standardize(reduce_dimension(X.sel(feature=c),
                                         operation="mean")
                        )
        Xs.append(x)
    for c in range(5, 13):
        # one-hot-encoded lancdover
        x = standardize(reduce_dimension(X.sel(feature=c),
                                         operation="sum")
                        )
        Xs.append(x)
    x = standardize(reduce_dimension(X.sel(feature=13),
                                     operation="std"))
    Xs.append(x)
    X_reduced = xr.concat(Xs, dim="new_dim")
    print(X_reduced)
    return X_reduced, Y



def split_dataset(X, Y, tvalsplit, testsplit,
                  cache=True, cachedir="./"):
    # split dataset
    nsamples = X.sizes["sample"]
    indices = np.arange(0, nsamples, 1)
    np.random.shuffle(indices)
    tval_indices = indices[0:int(nsamples*testsplit)]
    test_indices = indices[int(nsamples*testsplit)::]

    t_nsamples = tval_indices.shape[0]
    np.random.shuffle(tval_indices)
    train_indices = tval_indices[0:int(t_nsamples*tvalsplit)]
    vald_indices = tval_indices[int(t_nsamples*tvalsplit)::]

    X_train = X.isel(sample=train_indices)
    X_vald = X.isel(sample=vald_indices)
    X_test = X.isel(sample=test_indices)
    Y_train = Y.isel(sample=train_indices)
    Y_vald = Y.isel(sample=vald_indices)
    Y_test = Y.isel(sample=test_indices)

    print("training data: {0}".format(Y_train.sizes["sample"]))
    print("validation data: {0}".format(Y_vald.sizes["sample"]))
    print("test data: {0}".format(Y_test.sizes["sample"]))

    if cache:
        # save test data for later use
        save_dset(os.path.join(cachedir, "train.nc"), X_train, Y_train)
        save_dset(os.path.join(cachedir, "valid.nc"), X_vald, Y_vald)
        save_dset(os.path.join(cachedir, "test.nc"), X_test, Y_test)
    return [X_train, Y_train], [X_vald, Y_vald], [X_test, Y_test]


def save_dset(outpath, X, Y):
    dset = xr.Dataset({"features": X, "labels": Y})
    dset.to_netcdf(outpath)


def train_xgboost(X_train, Y_train, X_valid, Y_valid,
                  params=XGB_DEFAULT_PARAMS, num_round=10,
                  modelpath="best.model"):
    X_train_np = X_train.transpose("sample", ...).values
    Y_train_np = Y_train.transpose("sample", ...).values
    X_valid_np = X_valid.transpose("sample", ...).values
    Y_valid_np = Y_valid.transpose("sample", ...).values
    print(X_train_np.shape, Y_train_np.shape)
    dtrain = xgb.DMatrix(X_train_np, label=Y_train_np)
    dvalid = xgb.DMatrix(X_valid_np, label=Y_valid_np)
    evallist = [(dvalid, 'eval'), (dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_round, evallist)
    bst.save_model(modelpath)
    return bst    


def test_xgboost(bst, X_test, Y_test, acc_prob=0.5):
    X_test_np = X_test.transpose("sample", ...).values
    Y_test_np = Y_test.transpose("sample", ...).values
    dtest = xgb.DMatrix(X_test_np)
    ypred = bst.predict(dtest)
    # scores
    ypred_label = np.where(ypred>0.5, 1, 0)
    acc = accuracy_score(Y_test_np, ypred_label)
    fpr, tpr, _ = roc_curve(Y_test_np, ypred, pos_label=1)
    aucval = auc(fpr, tpr)
    print("test accuracy: {0:.3f}".format(acc))
    print("test auc: {0:.3f}".format(aucval))


def batch_train_test_split(ncpath, tvalsplit, testsplit,
                           use_cache=True, cachedir="./",
                           params=XGB_DEFAULT_PARAMS):
    """
    A simple batcher to perform a single training.
    """
    start = time.time()
    if use_cache:
        print("load cached datasets")
        train = load_netcdf(os.path.join(cachedir, "train.nc"))
        valid = load_netcdf(os.path.join(cachedir, "valid.nc"))
        test = load_netcdf(os.path.join(cachedir, "test.nc"))
    else:
        print("generate dataset")
        X, Y = make_dataset(ncpath)
        print_elapsed(start)
        print("split dataset")
        train, valid, test = split_dataset(X, Y,
                                           tvalsplit, testsplit,
                                           cachedir=cachedir)
        print_elapsed(start)
    print("start training")
    bst = train_xgboost(train[0], train[1], valid[0], valid[1])
    print_elapsed(start)
    print("evaluate results")
    test_xgboost(bst, test[0], test[1])
    print_elapsed(start)


def print_elapsed(start):
    print("elapsed: {0:.3f} [s]".format(time.time()-start))


if __name__ == "__main__":
    ncpath = "/mnt/d/data.nc"
    tvalsplit = 0.8
    testsplit = 0.8
    batch_train_test_split(ncpath, tvalsplit, testsplit, use_cache=False)