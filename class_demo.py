# Created by Andrew Silva on 10/11/18
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, explained_variance_score
import torch
import numpy as np
from pytorch_djinn.py_djinn import DJINNetwork



REGRESSION = False
if REGRESSION:
    d = datasets.load_boston()
    X = d.data
    Y = d.target
else:
    d = datasets.load_iris()
    X = d.data
    Y = d.target


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
np.random.seed(500)
torch.manual_seed(500)
model = DJINNetwork(num_trees=1, max_depth=4, drop_prob=0.0, do_regression=REGRESSION)

batch_size = 1
epochs = 30
lr = 1e-3

model.train(x_train, y_train, epochs, lr, batch_size)
# model.load_model()
preds = model.predict(x_test)

if REGRESSION:
    mse = mean_squared_error(y_test, preds)
    mabs = mean_absolute_error(y_test, preds)
    exvar = explained_variance_score(y_test, preds)
    print('MSE', mse)
    print('M Abs Err', mabs)
    print('Expl. Var.', exvar)
else:
    acc = accuracy_score(y_test, preds)
    print('Accuracy', acc)

epochs = 15
batch_size = 1
model.continue_training(x_train, y_train, epochs, lr, batch_size)
preds = model.predict(x_test)

if REGRESSION:
    mse = mean_squared_error(y_test, preds)
    mabs = mean_absolute_error(y_test, preds)
    exvar = explained_variance_score(y_test, preds)
    print('MSE', mse)
    print('M Abs Err', mabs)
    print('Expl. Var.', exvar)
else:
    acc = accuracy_score(y_test, preds)
    print('Accuracy', acc)


# make predictions
# m=model.predict(x_test) #returns the median prediction if more than one tree

# #evaluate results
# acc=sklearn.metrics.accuracy_score(y_test,m.flatten())
# print('Accuracy',acc)


