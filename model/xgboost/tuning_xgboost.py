import numpy as np
import xarray as xr
import xgboost as xgb
import time
import os
import json
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import train_xgboost as txg

model = xgb.XGBClassifier()
NCPATH = "/mnt/d/data.nc"
TVALSPLIT = 0.8
TESTSPLIT = 0.8
CACHEDIR = "./"

# load data
try:
    train = txg.load_netcdf(os.path.join(CACHEDIR, "train.nc"))
    valid = txg.load_netcdf(os.path.join(CACHEDIR, "valid.nc"))
    test = txg.load_netcdf(os.path.join(CACHEDIR, "test.nc"))
except(FileNotFoundError):
    print("generate dataset")
    X, Y = txg.make_dataset(NCPATH)
    print("split dataset")
    train, valid, test = txg.split_dataset(X, Y,
                                           TVALSPLIT, TESTSPLIT,
                                           cachedir=CACHEDIR)
X_train_np = train[0].transpose("sample", ...).values
Y_train_np = train[1].transpose("sample", ...).values
X_valid_np = valid[0].transpose("sample", ...).values
Y_valid_np = valid[1].transpose("sample", ...).values
X_train = np.concatenate([X_train_np, X_valid_np], axis=0)
Y_train = np.concatenate([Y_train_np, Y_valid_np], axis=0)

X_test = test[0].transpose("sample", ...).values
Y_test = test[1].transpose("sample", ...).values

# define parameters to tune
parameters = {"nthread":[4],
              "objective":["binary:logistic"],
              "learning_rate": [0.03, 0.05, 0.08],
              "max_depth": [3, 6, 9, 12, 15],
              "silent": [1],
              "subsample": np.arange(0.5, 1, 0.1).tolist(),
              "colsample_bytree": np.arange(0.5, 1, 0.1).tolist(),
              "n_estimators": [5, 10, 100, 1000], #number of trees, change it to 1000 for better results
              "lambda": [0.1, 1, 5, 10],
              "seed": [1337]}

# grid search cross validation
clf = GridSearchCV(model, parameters, n_jobs=5, 
                   cv=5, 
                   scoring="accuracy",
                   verbose=2, refit=True)
clf.fit(X_train, Y_train)
best_parameters = clf.best_params_
print('Raw Acuracy score:', clf.score(X_train, Y_train))
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
with open("best_params.json", "w") as f:
    json.dump(best_parameters, f)
with open("tuned_model.pickle", "wb") as f:
    pickle.dump(clf, f)
y_true, y_pred = Y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
