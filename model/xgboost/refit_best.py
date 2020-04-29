import numpy as np
import xarray as xr
import xgboost as xgb
import seaborn
import time
import json
import os
from sklearn.metrics import classification_report, roc_curve, auc
from matplotlib import pyplot as plt
import train_xgboost as txg
seaborn.set_style("whitegrid")


NCPATH = "/mnt/d/data.nc"
TVALSPLIT = 0.8
TESTSPLIT = 0.8
CACHEDIR = "./"

# load parameters
with open("best_params.json", "r") as f:
    params = json.load(f)

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

# re-fit the model
dtrain = xgb.DMatrix(X_train, label=Y_train)
bst = xgb.train(params, dtrain, params["n_estimators"])
bst.save_model("best.xgbmodel")

# test
dtest = xgb.DMatrix(X_test)
ypred = bst.predict(dtest)
# scores
ypred_label = np.where(ypred>0.5, 1, 0)
fpr, tpr, _ = roc_curve(Y_test, ypred, pos_label=1)
aucval = auc(fpr, tpr)
print("test auc: {0:.3f}".format(aucval))
print(classification_report(Y_test, ypred_label))
# draw a plot of importance
xgb.plot_importance(bst)
if not os.path.exists("./images"):
    os.makedirs("./images")
plt.savefig("./images/importance.png", dpi=300)
plt.close()
# draw a plot of graph
xgb.plot_tree(bst, num_trees=6)
plt.savefig("./images/tree.png", dpi=300)
plt.close()
