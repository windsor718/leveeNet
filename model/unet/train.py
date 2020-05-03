import argparse
import configparser
import pickle
import glob
import os
import image_generator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from model import u_Net

parser = argparse.ArgumentParser(description="Train levee detection model")
parser.add_argument("-c", "--config", help="configuration file",
                    type=str, required=True)
parser.add_argument("--usecache", help="use cached files and skip preprocess",
                    action="store_true", default=False)

# parse args
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config)

# load parameters
n_classes = config.getint("model", "n_classes")
model_outpath = config.get("model", "model_outpath")
weight_outpath = config.get("model", "weight_outpath")
resize_v = config.getint("generator", "resize_v")
resize_h = config.getint("generator", "resize_h")
n_channels = config.getint("generator", "n_channels")
max_pool = config.getint("generator", "max_pool")
shuffle = config.getboolean("generator", "shuffle")
augment = config.getboolean("generator", "augment")
batch_size = config.getint("train", "batch_size")
num_epochs = config.getint("train", "num_epochs")
log_path = config.get("train", "log_path")
history_outpath = config.get("train", "history_outpath")
data_path = config.get("data", "data_path")

# check data types
assert isinstance(n_classes, int), "n_classes must be int, but got {0}".format(type(n_classes))
assert isinstance(resize_v, int), "resize_v must be tuple, but got {0}".format(type(resize_v))
assert isinstance(resize_h, int), "resize_h must be tuple, but got {0}".format(type(resize_h))
assert isinstance(n_channels, int), "resize_h must be tuple, but got {0}".format(type(n_channels))
assert isinstance(batch_size, int), "batch_size must be int, but got {0}".format(type(batch_size))
assert (isinstance(max_pool, int) | (max_pool is None)), "max_pool must be None or int, but got {0}".format(type(max_pool))
assert isinstance(num_epochs, int), "num_epoch must be int, but got {0}".format(type(num_epochs))

# additional variable definitions and mkdir
image_size = (resize_v, resize_h, n_channels)
trainpath = os.path.join(os.path.dirname(data_path), "train")
validpath = os.path.join(os.path.dirname(data_path), "valid")
testpath = os.path.join(os.path.dirname(data_path), "test")
for path in [trainpath, validpath, testpath,
             log_path, model_outpath, weight_outpath]:
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(path)

trainfiles = glob.glob(trainpath + "/*")
validfiles = glob.glob(validpath + "/*")
testfiles = glob.glob(testpath + "/*")

# instantiate generator
train_generator = image_generator.DataGenerator(trainfiles,
                                                n_classes, batch_size,
                                                image_size, max_pool,
                                                shuffle, augment)
valid_generator = image_generator.DataGenerator(validfiles,
                                                n_classes, batch_size,
                                                image_size, max_pool,
                                                shuffle, augment)

# callbacks
# reduces learning rate if no improvement are seen
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0000001)

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
model = u_Net(n_classes, image_size)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(), metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=log_path, histogram_freq=1)

history = model.fit(train_generator,
                    validation_data=valid_generator,
                    shuffle=True,
                    epochs=num_epochs,
                    steps_per_epoch=len(train_generator),
                    validation_steps=len(valid_generator),
                    callbacks=[tensorboard, early_stop, checkpoint,
                               learning_rate_reduction],
                    verbose=1)
model.save(model_outpath)
with open(history_outpath, "wb") as f:
    pickle.dump(history, f)

#
# testdata = xr.open_dataset(testpath)
X_test, Y_test = \
    image_generator.DataGenerator(testfiles,
                                  n_classes, batch_size,
                                  image_size, max_pool,
                                  shuffle, augment).testDataGenerator()
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy;', score[1])
