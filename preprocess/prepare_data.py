import numpy as np
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


def batch_process(srcDir, parallel=True):
    """
    batch the image processing in either serial or parallel.

    Args:
        srcDir (str): image source directory
        parallel (bool)
    
    Returns:
        np.ndarray: features
        np.ndarray: labels
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
        outitems = process_images(files, verbose=True)
        X = outitems[0]
        Y = outitems[1]
    print("Feature matrix shape [samples, features, d0, d1]: ", X.shape)
    print("Label shape [samples]", Y.shape)
    # featureWiseStandardization for R, G, B, NIR, SWIR
    for i in range(0, 5):
        X[:, i, :, :] = pp.featureWiseStandardization(X[:, i, :, :])
    return X, Y


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
    X_all = np.vstack(Xlist)  # list of arrays
    Y_all = np.array(Ylist).astype(DTYPE)  # list of scalars
    return [X_all, Y_all]


def process_image(tiffpath):
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
    y = get_label(farray[7])
    print(X.shape)
    X = X.astype(DTYPE)
    return X, y


def get_label(leveeArray, threshold=10):
    """
    count number of levee pixels and return label for a image.

    Args:
        leveeArray (numpy.ndarray): levee location array
            (1 for levee, 0/nan for non-levee)
        threshold (int): minimum number of levee pixels to label as levee image
    """
    leveeArray[np.isnan(leveeArray)] = 0
    leveeCount = np.sum(leveeArray)
    if leveeCount > threshold:
        print("levee")
        return 1
    else:
        return 0


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


def save_to_hdf5(h5path, X, Y, X_attr, Y_attr):
    t = datetime.datetime.now()
    f = h5py.File(h5path, mode="w")
    labels = f.create_dataset("labels", data=Y)
    for key, item in Y_attr.items():
        labels.attr[key] = item
    labels.attrs["creationDate"] = t.strftime("%Y%m%d_%H:%M")
    features = f.create_dataset("features", data=X)
    for key, item in X_attr.items():
        features.attr[key] = item
    features.attrs["creationDate"] = t.strftime("%Y%m%d_%H:%M")
    f.close()


def describe_dataset(X, Y):
    """
    print out summary of dataset.

    Args:
        X (numpy.ndarray): feature array
        Y (numpy.ndarray): label array
    
    Returns:
        None
    """
    uniques, counts = np.unique(a, return_counts=True)
    print("number of labels:")
    for idx, u in enumerate(uniques):
        print("\t{0}:{1}/{2}".format(u, counts[idx], Y.shape[0]))
    num_of_bands = X.shape[1]
    print("feature stats:")
    for i in range(num_of_bands):
        data = X[:, i, :, :]
        "feature_{0}: min {1}; max{2}; mean{3}".format(data.min(), data.max(), data.mean())


if __name__ == "__main__":
    X, Y = batch_process("../Dataset/images/", parallel=False)
    save_to_hdf5("../Dataset/data.hdf5", X, Y, X_ATTR, Y_ATTR)
    describe_dataset(X, Y)