import tensorflow
import numpy as np
import xarray as xr
import os
from tensorflow.keras import utils
from imgaug import augmenters as iaa


class DataGenerator(tensorflow.keras.utils.Sequence):
    """
    custom data generator to perform following:
        1. generate a batch of numpy.ndarray from xarray.DataArray
        2. if enabled, perform maxPooling to reduce image size
        2. if enabled, perform data augumentation:
            - horizontally flip 50% of all images
            - vertically flip 20% of all images
            - rotate by -45 to +45 degrees for 50% of all images

    Args:
        files (list): list of data file paths
        batch_size (int): number of images to load in a batch
        image_size (tuple): (int v, int h). output image shape
        max_pool (int): default None. if int, then pool image via max pooling.
            This will reduce image size by 1/max_pool
        shuffle (bool): if True, shuffle indexes every batch.
            in either way the indexes are shuffled when you make an instance.
        augment (bool): if True, perform data augmentation.
        plot (bool): if True, plot loss at the end of the epoch.
    """
    def __init__(self, files, num_classes,
                 batch_size=32, image_size=(256, 256), max_pool=None,
                 shuffle=False, augment=False):
        self.files = files
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_pool = max_pool
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end(True)  # shuffle at the very beggining

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.files) / self.batch_size))

    def on_epoch_end(self, shuffle):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.files))
        if shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        "Generate one batch of data"
        # selects indices of data for next batch
        indexes = \
            self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # select data and load images; X: features, Y: segmentation maps
        X_batch, Y_batch = self.load_files(indexes)
        # preprocess and augment data
        if isinstance(self.max_pool, int):
            X_batch = self.maxPooling(X_batch, self.max_pool)
            Y_batch = self.maxPooling(Y_batch, self.max_pool)
        resizer = iaa.Resize({"height": self.image_size[0],
                              "width": self.image_size[1]},
                             interpolation="nearest")
        X_batch = np.stack(resizer.augment_images(X_batch), axis=0)
        Y_batch = np.stack(resizer.augment_images(Y_batch), axis=0)

        if self.augment:
            X_batch, Y_batch = self.augmentor(X_batch,
                                              np.expand_dims(Y_batch, axis=3))
            X_batch = np.stack(X_batch, axis=0)
            Y_batch = np.stack(Y_batch, axis=0)

        return X_batch, Y_batch, [None]

    def load_files(self, indexes):
        X_batch = []
        Y_batch = []
        for idx in indexes:
            data = xr.open_dataset(self.files[idx])
            featureArray = data["features"].transpose(..., "feature")
            features = np.expand_dims(featureArray.values, axis=0)
            labels = np.expand_dims(data["labels"].values, axis=0)
            X_batch.append(features)
            Y_batch.append(labels)
        X_batch = np.concatenate(X_batch, axis=0)
        Y_batch = np.concatenate(Y_batch, axis=0)
        return X_batch, Y_batch

    def maxPooling(self, X, kernel_size):
        aug = iaa.MaxPooling(kernel_size, keep_size=False)
        return aug.augment_images(X)

    def augmentor(self, X, Y):
        "Apply data augmentation"
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
                  [
                   # apply the following augmenters to most images
                   iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                   iaa.Flipud(0.2),  # vertically flip 20% of all images
                   sometimes(iaa.Affine(
                      rotate=(-10, 10),  # rotate by -45 to +45 degrees
                      order=0,  # use nearest neighbour
                      mode="constant"  # 0-padding (cv2.BORDER_CONSTANT)
                   ))
                  ],
                  random_order=True  # apply augmenters in random order
              )
        images_aug, segmaps_aug = seq(images=X, segmentation_maps=Y)
        return images_aug, segmaps_aug

    def testDataGenerator(self):
        # preprocess and augment data
        X = self.X.values
        Y = self.Y.values
        if isinstance(self.max_pool, int):
            X_batch = self.maxPooling(X, self.max_pool)
        X_batch = iaa.Resize({"height": self.image_size[0],
                              "width": self.image_size[1]},
                             interpolation="nearest").augment_images(X_batch)
        X_batch = np.stack(X_batch, axis=0)

        if self.augment:
            X_batch = self.augmentor(X_batch)
            X_batch = np.stack(X_batch, axis=0)

        # categorical label for Keras
        Y_batch = utils.to_categorical(Y, max(2, self.num_classes))
        return X_batch, Y_batch


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
    return (X_darray.isel(sample=sampled_indices),
            Y_darray.isel(sample=sampled_indices))


def split_cache(X, Y, rootpath):
    filepaths = []
    nsamples = X.sizes["sample"]
    samples_nums = X["sample"].values.tolist()
    for idx in range(nsamples):
        features = X.isel(sample=idx)
        label = Y.isel(sample=idx)
        sample_idx = samples_nums[idx]
        dset = xr.Dataset({"features": features,
                           "labels": label})
        opath = os.path.join(rootpath,
                             "{0:05d}.nc".format(sample_idx))
        dset.to_netcdf(opath)
        filepaths.append(opath)
    return filepaths
