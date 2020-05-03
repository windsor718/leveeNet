import numpy as np
import xarray as xr
import argparse
import configparser
import pickle
import glob
import os
import image_generator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from model import leveeNet

parser = argparse.ArgumentParser(description="Train levee detection model")
parser.add_argument("-c", "--config", help="configuration file",
                    type=str, required=True)
parser.add_argument("--usecache", help="use cached files and skip preprocess",
                    action="store_true", default=False)

# parse args and check data types
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config)

n_classes = config.getint("model", "n_classes")
assert isinstance(n_classes, int), "n_classes must be int, but got {0}".format(type(n_classes))
model_outpath = config.get("model", "model_outpath")
weight_outpath = config.get("model", "weight_outpath")

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
num_epochs = config.getint("train", "num_epochs")
assert isinstance(num_epochs, int), "num_epoch must be int, but got {0}".format(type(num_epochs))
logpath = config.get("train", "log_path")
history_outpath = config.get("train", "history_outpath")
testsplit = config.getfloat("train", "testsplit")
validsplit = config.getfloat("train", "validsplit")

datapath = config.get("data", "data_path")
trainpath = os.path.join(os.path.dirname(datapath), "split_train")
validpath = os.path.join(os.path.dirname(datapath), "split_valid")
testpath = os.path.join(os.path.dirname(datapath), "split_test")
for path in [trainpath, validpath, testpath]:
    if not os.path.exists(path):
        os.makedirs(path)


# read data as DataArray
if not args.usecache:
    print("read dataset...")
    data = xr.open_dataset(datapath)
    X = data["features"]
    Y = data["labels"]
    print("read dataset done.")

    # down sample dataset to balance number of data in classes
    print("down-sampled:")
    X, Y = image_generator.match_nsamples(X, Y)

    # split dataset
    print("split dataset")
    nsamples = X.sizes["sample"]
    indices = np.arange(0, nsamples, 1)
    np.random.shuffle(indices)
    tval_indices = indices[0:int(nsamples*testsplit)]
    test_indices = indices[int(nsamples*testsplit)::]

    np.random.shuffle(tval_indices)
    nsamples_t = tval_indices.shape[0]
    train_indices = tval_indices[0:int(nsamples_t*validsplit)]
    valid_indices = tval_indices[int(nsamples_t*validsplit)::]

    X_train = X.isel(sample=train_indices)
    X_valid = X.isel(sample=valid_indices)
    X_test = X.isel(sample=test_indices)
    Y_train = Y.isel(sample=train_indices)
    Y_valid = Y.isel(sample=valid_indices)
    Y_test = Y.isel(sample=test_indices)

    print("Training: {0}".format(int(nsamples_t*testsplit)))
    print("Validation: {0}".format(nsamples_t - int(nsamples_t*testsplit)))
    print("Test: {0}".format(nsamples - int(nsamples*testsplit)))

    # split data for faster access from image_generator
    print("cache data")
    trainfiles = image_generator.split_cache(X_train, Y_train,
                                             trainpath)
    validfiles = image_generator.split_cache(X_valid, Y_valid,
                                             validpath)
    testfiles = image_generator.split_cache(X_test, Y_test,
                                            testpath)
else:
    trainfiles = glob.glob(trainpath + "/*")
    validfiles = glob.glob(validpath + "/*")
    testfiles = glob.glob(testpath + "/*")

# instantiate generator
print("start learning")
train_generator = image_generator.DataGenerator(trainfiles,
                                                n_classes, batch_size,
                                                image_size, max_pool,
                                                shuffle, augment)
valid_generator = image_generator.DataGenerator(validfiles,
                                                n_classes, batch_size,
                                                image_size, max_pool,
                                                shuffle, augment)
print(train_generator[0])
# callbacks
# reduces learning rate if no improvement are seen
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
#                                             patience=2,
#                                             verbose=1,
#                                             factor=0.5,
#                                             min_lr=0.0000001)

# stop training if no improvements are seen
early_stop = EarlyStopping(monitor="val_loss",
                           mode="min",
                           patience=5,
                           restore_best_weights=True)

# saves model weights to filae
checkpoint = ModelCheckpoint(weight_outpath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             save_weights_only=True)

# start session
model = leveeNet(n_classes, image_size)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(), metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=logpath, histogram_freq=1)

history = model.fit(train_generator,
                    validation_data=valid_generator,
                    shuffle=True,
                    epochs=num_epochs,
                    steps_per_epoch=len(train_generator),
                    validation_steps=len(valid_generator),
                    callbacks=[tensorboard, early_stop, checkpoint],
                    # use_multiprocessing=True,
                    # workers=6,
                    verbose=1)
model.save(model_outpath)
with open(history_outpath, "wb") as f:
    pickle.dump(history, f)

#
# testdata = xr.open_dataset(testpath)
X_test = data["features"]
Y_test = data["labels"]
X_test, Y_test = image_generator.DataGenerator(testfiles,
                                               n_classes, batch_size,
                                               image_size, max_pool,
                                               shuffle, augment).testDataGenerator()
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy;', score[1])
