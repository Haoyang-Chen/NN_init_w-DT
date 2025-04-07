# Created by Andrew Silva on 10/9/18

from sklearn.tree import DecisionTreeClassifier
from py_djinn_utils_EXAI import tree_to_nn_weights, net_dropout_regression
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
import torch
import numpy as np

use_gpu = torch.cuda.is_available()


class DJINNetwork:
    """
    DJINN model in pytorch
    """

    def __init__(self,max_depth=4):
        self.max_depth = max_depth
        self.tree_network = None
        self.model_fn = 'tree.pth.tar'

    def train(self, input_data, labels, epochs=10, lr=1e-3, batch_size=32):
        """
        :param input_data: x_in data
        :param labels: y_in data (y_out maybe?)
        :param epochs: number of epochs to train for,default 10
        :param lr: learning rate
        :param batch_size: batch size
        :return: ?
        """

        # fit rf to regression or classification
        clf = DecisionTreeClassifier(max_depth=self.max_depth)
        clf.fit(input_data, labels)

        # ensure rf is valid
        if clf.tree_.max_depth <= 1:
            print("RF failed")
            return
        # Map trees to neural networks
        tree_to_network = tree_to_nn_weights(input_data, labels, clf)
        tree_net_arr = net_dropout_regression(tree_to_network,
                                              input_data,
                                              labels,
                                              lr,
                                              epochs,
                                              batch_size)
        self.tree_network = tree_net_arr

    def forward(self, x):
        """
        Forward pass through the network
        :param x: input data
        :return: output of the network
        """
        if self.tree_network is None:
            raise ValueError("Model not trained yet.")
        # Pass through each tree in the ensemble
        return self.tree_network(x)


d = datasets.load_iris()
X = d.data
Y = d.target


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
np.random.seed(500)
torch.manual_seed(500)
model = DJINNetwork(max_depth=4)

batch_size = 1
epochs = 30
lr = 1e-3

model.train(x_train, y_train, epochs, lr, batch_size)
# model.load_model()
preds = model.predict(x_test)

acc = accuracy_score(y_test, preds)
print('Accuracy', acc)
