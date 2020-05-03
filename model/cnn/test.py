import numpy as np
import xarray as xr
import argparse
import configparser
import pickle
import image_generator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from model import leveeNet


parser = argparse.ArgumentParser(description="Train levee detection model")
parser.add_argument("-c", "--config", help="configuration file",
                    type=str, required=True)
parser.add_argument("-w", "--weights", help="weight hdf5 file path",
                    type=str, required=True)

# parse args and check data types
args = parser.parse_args()
weight_path = args.weights
config = configparser.ConfigParser()
config.read(args.config)
n_classes = config.getint("model", "n_classes")
assert isinstance(n_classes, int), "n_classes must be int, but got {0}".format(type(n_classes))

resize_v = config.getint("generator", "resize_v")
assert isinstance(resize_v, int), "resize_v must be tuple, but got {0}".format(type(resize_v))
resize_h = config.getint("generator", "resize_h")
assert isinstance(resize_h, int), "resize_h must be tuple, but got {0}".format(type(resize_h))
n_channels = config.getint("generator", "n_channels")
assert isinstance(n_channels, int), "resize_h must be tuple, but got {0}".format(type(n_channels))
image_size = (resize_v, resize_h, n_channels)

max_pool = config.getint("generator", "max_pool")
assert (isinstance(max_pool, int) | (max_pool is None)), "max_pool must be None or int, but got {0}".format(type(max_pool))

shuffle = config.getboolean("generator", "shuffle")
augment = config.getboolean("generator", "augment")

batch_size = config.getint("train", "batch_size")
assert isinstance(batch_size, int), "batch_size must be int, but got {0}".format(type(batch_size))

testpath = config.get("data", "test_path")

# load data
data = xr.open_dataset(testpath)
X_test = data["features"]
Y_test = data["labels"]
X_test, Y_test = image_generator.DataGenerator(X_test, Y_test,
                                               n_classes, batch_size,
                                               image_size, max_pool,
                                               shuffle, augment).testDataGenerator()

model = leveeNet(n_classes, image_size)
model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
model.load_weights(weight_path)
score = model.evaluate(X_test, Y_test, verbose=0)
print("Test score:", score[0])
print("Test accuracy;", score[1])
