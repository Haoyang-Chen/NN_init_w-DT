# Created by Andrew Silva on 10/9/18
import torch
import numpy as np
from scipy import stats
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from py_djinn_utils import tree_to_nn_weights, net_dropout_regression, \
    save_checkpoint, continue_training_trees, load_checkpoint

use_gpu = torch.cuda.is_available()


class DJINNetwork:
    """
    DJINN model in pytorch
    """

    def __init__(self, num_trees=1, max_depth=4, drop_prob=0.0, do_regression=True):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.drop_prob = drop_prob
        self.do_regression = do_regression
        self.single_tree_network = None
        self.model_fn = 'all_trees.pth.tar'

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
        if self.do_regression:
            clf = RandomForestRegressor(self.num_trees, max_depth=self.max_depth)
        else:
            clf = RandomForestClassifier(self.num_trees, max_depth=self.max_depth)
        clf.fit(input_data, labels)

        # ensure rf is valid
        if clf.estimators_[0].tree_.max_depth <= 1:
            print("RF failed")
            return
        # Map trees to neural networks
        tree_to_network = tree_to_nn_weights(self.do_regression, input_data, labels, self.num_trees, clf)
        tree_net_arr = net_dropout_regression(self.do_regression,
                                              tree_to_network,
                                              input_data,
                                              labels,
                                              lr,
                                              epochs,
                                              batch_size,
                                              self.drop_prob)
        self.single_tree_network = tree_net_arr
        save_checkpoint(self.single_tree_network, self.model_fn)

    def predict(self, input_data, return_percentiles=False):
        preds = []
        for tree_net in self.single_tree_network:
            tree_net.eval()
            if use_gpu:
                tree_net = tree_net.cuda()
            net_in = torch.autograd.Variable(torch.Tensor(input_data))
            if use_gpu:
                net_in = net_in.cuda()
            pred = tree_net(net_in)
            pred = torch.argmax(pred, dim=1)
            if use_gpu:
                pred = pred.cpu()
            preds.append(pred.detach().numpy())

        preds = np.array(preds).transpose()

        middle = np.percentile(preds, 50, axis=1)
        lower = np.percentile(preds, 25, axis=1)
        upper = np.percentile(preds, 75, axis=1)
        if return_percentiles:
            return [lower, middle, upper]
        else:
            return stats.mode(preds, axis=1)[0].reshape(-1)  # Take the majority vote. This will fail with regression
            # return middle

    def continue_training(self, input_data, labels, epochs=10, lr=1e-3, batch_size=32):
        if self.single_tree_network is None:
            self.load_model()
        self.single_tree_network = continue_training_trees(self.single_tree_network, input_data, labels,
                                                           lr=lr, batch_size=batch_size, num_epochs=epochs,
                                                           regression=self.do_regression)
        save_checkpoint(self.single_tree_network, self.model_fn)

    def load_model(self):
        if os.path.exists(self.model_fn):
            self.single_tree_network = load_checkpoint(self.model_fn)
        else:
            print("No model available")
            return -1
        return 0
