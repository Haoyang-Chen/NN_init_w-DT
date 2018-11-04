# Created by Andrew Silva on 10/9/18
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

use_gpu = torch.cuda.is_available()


def save_checkpoint(djinn_model, filename='tree0.pth.tar'):
    """
    Helper function to save checkpoints of PyTorch models
    :param state: Everything to save with the checkpoint (params/weights, optimizer state, epoch, loss function, etc.)
    :param filename: Filename to save the checkpoint under
    :return: None
    """
    if type(djinn_model) is not list:
        djinn_model = [djinn_model]
    master_list = []
    for model in djinn_model:
        # state_dict = {}
        # for layer_number, layer in enumerate(model.layers):
        #     for ele in layer.state_dict():
        #         if 'weight' in ele:
        #             key_name = 'weight'+str(layer_number)
        #         elif 'bias' in ele:
        #             key_name = 'bias' + str(layer_number)
        #         else:
        #             key_name = ele + str(layer_number)
        #         state_dict[key_name] = layer.state_dict()[ele]
        # for key in model.state_dict():
        #     state_dict[key] = model.state_dict()[key]
        state_dict = model.state_dict()
        master_list.append(state_dict)
    torch.save(master_list, filename)


def load_checkpoint(filename='tree0.pth.tar', drop_prob=0.0):
    if not use_gpu:
        model_checkpoint_master = torch.load(filename, map_location='cpu')
    else:
        model_checkpoint_master = torch.load(filename)
    master_model_list = []
    for model_checkpoint in model_checkpoint_master:
        num_layers = len(model_checkpoint.keys())//2
        fake_weights = []
        for i in range(num_layers):
            fake_weights.append(np.random.random_sample([4, 4]))
        new_net = PyDJINN(5, fake_weights, fake_weights)
        for index in range(0, num_layers-1):
            # weight_key = 'weight'+str(index)
            # bias_key = 'bias'+str(index)
            weight_key_value_pair = model_checkpoint.popitem(last=False)
            bias_key_value_pair = model_checkpoint.popitem(last=False)
            new_layer = nn.Linear(4, 4)
            new_layer.weight.data = weight_key_value_pair[1]
            new_layer.bias.data = bias_key_value_pair[1]
            new_seq = nn.Sequential(
                new_layer,
                nn.ReLU(),
                nn.Dropout(drop_prob)
            )
            new_net.layers[index] = new_seq
        new_net.final_layer.weight.data = model_checkpoint['final_layer.weight']
        new_net.final_layer.bias.data = model_checkpoint['final_layer.bias']
        master_model_list.append(new_net)
    return master_model_list

def xavier_init(dim_in, dim_out):
    dist = np.random.normal(0.0, scale=np.sqrt(3.0/(dim_in+dim_out)))
    return dist


def tree_to_nn_weights(regression, x_in, y_in, num_trees, regressor):
    """
    :param regression: flag, regression or not
    :param x_in: input data (batch first)
    :param y_in: output data (batch first)
    :param num_trees: num trees
    :param regressor: random forest regressor object
    :return:
    """
    dim_in = x_in.shape[1]
    if regression:
        if y_in.size > y_in.shape[0]:
            dim_out = y_in.shape[1]
        else:
            dim_out = 1
    else:
        dim_out = len(np.unique(y_in))

    tree_to_net = {
        'input_dim': dim_in,
        'output_dim': dim_out,
        'net_shape': {},
        'weights': {},
        'biases': {}
    }

    for tree in range(num_trees):
        tree_in = regressor.estimators_[tree].tree_
        features = tree_in.feature
        num_nodes = tree_in.node_count
        children_left = tree_in.children_left
        children_right = tree_in.children_right

        node_depth = np.zeros(num_nodes, dtype=np.int64)
        is_leaves = np.zeros(num_nodes, dtype=np.int64)
        stack = [(0, -1)]  # node id and parent depth of the root (0th id, no parents means -1...)
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], parent_depth+1))
                stack.append((children_right[node_id], parent_depth+1))
            else:
                is_leaves[node_id] = 1

        node_dict = {}
        for i in range(len(features)):
            node_dict[i] = {}
            node_dict[i]['depth'] = node_depth[i]
            if features[i] >= 0:
                node_dict[i]['features'] = features[i]
            else:
                node_dict[i]['features'] = -2
            node_dict[i]['child_left'] = features[children_left[i]]
            node_dict[i]['child_right'] = features[children_right[i]]

        num_layers = len(np.unique(node_depth))
        nodes_per_level = np.zeros(num_layers)
        leaves_per_level = np.zeros(num_layers)

        for i in range(num_layers):
            ind = np.where(node_depth == i)[0]
            nodes_per_level[i] = len(np.where(features[ind] >= 0)[0])
            leaves_per_level[i] = len(np.where(features[ind] < 0)[0])

        max_depth_feature = np.zeros(x_in.shape[1])
        for i in range(len(max_depth_feature)):
            ind = np.where(features == i)[0]
            if len(ind) > 0:
                max_depth_feature[i] = np.max(node_depth[ind])

        djinn_arch = np.zeros(num_layers, dtype=np.int64)

        djinn_arch[0] = dim_in
        for i in range(1, num_layers):
            djinn_arch[i] = djinn_arch[i-1] + nodes_per_level[i]
        djinn_arch[-1] = dim_out

        djinn_weights = {}
        for i in range(num_layers-1):
            djinn_weights[i] = np.zeros((djinn_arch[i+1], djinn_arch[i]))

        new_indices = []
        for i in range(num_layers-1):
            input_dim = djinn_arch[i]
            output_dim = djinn_arch[i+1]
            new_indices.append(np.arange(input_dim, output_dim))
            for f in range(dim_in):
                if i < max_depth_feature[f]-1:
                    djinn_weights[i][f, f] = 1.0
            input_index = 0
            output_index = 0
            for index, node in node_dict.items():
                if node['depth'] != i or node['features'] < 0:
                    continue
                feature = node['features']
                left = node['child_left']
                right = node['child_right']
                if index == 0 and (left < 0 or right < 0):
                    for j in range(i, num_layers-2):
                        djinn_weights[j][feature, feature] = 1.0
                    djinn_weights[num_layers-2][:, feature] = 1.0
                if left >= 0:
                    if i == 0:
                        djinn_weights[i][new_indices[i][input_index],
                                         feature] = xavier_init(input_dim, output_dim)
                    else:
                        djinn_weights[i][new_indices[i][input_index],
                                         new_indices[i-1][output_index]] = xavier_init(input_dim, output_dim)

                    djinn_weights[i][new_indices[i][input_index], left] = xavier_init(input_dim, output_dim)
                    input_index += 1
                    # TODO: comment below?
                    if output_index >= len(new_indices[i-1]):
                        output_index = 0

                if left < 0 and index != 0:
                    leaf_ind = new_indices[i-1][output_index]
                    for j in range(i, num_layers-2):
                        djinn_weights[j][leaf_ind, leaf_ind] = 1.0
                    djinn_weights[num_layers-2][:, leaf_ind] = 1.0

                if right >= 0:
                    if i == 0:
                        djinn_weights[i][new_indices[i][input_index],
                                         feature] = xavier_init(input_dim, output_dim)
                    else:
                        djinn_weights[i][new_indices[i][input_index],
                                         new_indices[i-1][output_index]] = xavier_init(input_dim, output_dim)

                    djinn_weights[i][new_indices[i][input_index], right] = xavier_init(input_dim, output_dim)
                    input_index += 1
                    # TODO: comment below?
                    if output_index >= len(new_indices[i-1]):
                        output_index = 0

                if right < 0 and index != 0:
                    leaf_ind = new_indices[i-1][output_index]
                    for j in range(i, num_layers-2):
                        djinn_weights[j][leaf_ind:leaf_ind] = 1.0
                    djinn_weights[num_layers-2][:, leaf_ind] = 1.0
                output_index += 1
                # TODO: uncomment below?
                # if output_index >= len(new_indices[i - 1]):
                #     output_index = 0

        m = len(new_indices[-2])
        ind = np.where(abs(djinn_weights[num_layers-3][:, -m:]) > 0)[0]
        for indices in range(len(djinn_weights[num_layers-2][:, ind])):
            djinn_weights[num_layers-2][indices, ind] = xavier_init(input_dim, output_dim)

        tree_name = 'tree_' + str(tree)
        tree_to_net['net_shape'][tree_name] = djinn_arch
        tree_to_net['weights'][tree_name] = djinn_weights
        tree_to_net['biases'][tree_name] = []  # biases?

    return tree_to_net


class PyDJINN(nn.Module):
    def __init__(self, input_dim, weights, biases, drop_prob=0.5):
        super(PyDJINN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, weights[0].shape[0]))
        weight_inits = torch.Tensor(weights[0])
        weight_inits.requires_grad = True
        self.layers[0].weight.data = weight_inits
        # self.layers[0].bias.data.fill_(biases[0])
        last_dim = weights[0].shape[0]
        for index in range(1, len(weights)-1):
            new_linear_layer = nn.Linear(last_dim, weights[index].shape[0])
            weight_inits = torch.Tensor(weights[index])
            weight_inits.requires_grad = True
            new_linear_layer.weight.data = weight_inits
            # new_linear_layer.bias.data.fill_(biases[index])
            new_layer = nn.Sequential(
                new_linear_layer,
                nn.ReLU(),
                nn.Dropout(drop_prob)
            )
            self.layers.append(new_layer)
            last_dim = weights[index].shape[0]
        self.final_layer = nn.Linear(last_dim, weights[-1].shape[0])
        weight_inits = torch.Tensor(weights[-1])
        weight_inits.requires_grad = True
        self.final_layer.weight.data = weight_inits
        # self.final_layer.bias.data.fill_(biases['out'])

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
            # print(input_data)
        return self.final_layer(input_data)


def net_dropout_regression(regression, tree_dict, x_in, y_in,
                           learning_rate, num_epochs, batch_size, drop_prob, log_every=5):
    """ Trains neural networks in PyTorch, given initial weights from decision tree.
        :param regression: do regression or classification (true for regression)
        :param tree_dict: from above function
        :param x_in : training data samples
        :param y_in : training data labels
        :param learning_rate : learning rate
        :param num_epochs : epochs to train for
        :param batch_size : batch size
        :param drop_prob : for dropout in most hidden layers
        :param log_every: epoch log interval
    """
    input_dim = tree_dict['input_dim']


    tree_number = 0
    tree_list = []
    for keys in tree_dict["weights"].keys():

        npl = tree_dict['net_shape'][keys]
        n_hidden = {}
        for i in range(1, len(npl) - 1):
            n_hidden[i] = npl[i]

        # existing weights from above, biases could be improved... ignoring for now
        w = []
        b = []
        for i in range(0, len(tree_dict['net_shape'][keys]) - 1):
            w.append(tree_dict['weights'][keys][i].astype(np.float32))
            # b.append(xavier_init(n_hidden[i], n_hidden[i+1]))  # This might be wrong
        tree_net = PyDJINN(input_dim, weights=w, biases=b, drop_prob=drop_prob)
        # prediction is the output from the MLP
        if regression:
            loss = nn.MSELoss()
        else:
            loss = nn.CrossEntropyLoss()
        if use_gpu:
            loss = loss.cuda()
            tree_net = tree_net.cuda()

        for epoch in range(num_epochs):
            x_train, x_test, y_train, y_test = train_test_split(x_in, y_in, test_size=0.1)
            # want to maintain class label indices. i think.
            # if not regression:
            #     y_train = np.equal.outer(y_train.flatten(), np.arange(len(np.unique(y_train)))).astype("float32")
            #     y_test = np.equal.outer(y_test.flatten(), np.arange(len(np.unique(y_test)))).astype("float32")

            learning_rate_ep = learning_rate
            if epoch > num_epochs/2:
                learning_rate_ep *= 0.1
            opt = torch.optim.Adam(tree_net.parameters(), lr=learning_rate_ep)

            epoch_train_loss = 0
            for ind in range(0, len(x_train), batch_size):
                samples = Variable(torch.Tensor(x_train[ind:ind+batch_size]))
                if regression:
                    labels = Variable(torch.Tensor(y_train[ind:ind + batch_size]))
                else:
                    labels = Variable(torch.LongTensor(y_train[ind:ind+batch_size]))
                if use_gpu:
                    samples = samples.cuda()
                    labels = labels.cuda()
                preds = tree_net(samples)
                if regression:
                    preds = preds.view(-1)
                iter_loss = loss(preds, labels)

                opt.zero_grad()
                iter_loss.backward()
                opt.step()
                epoch_train_loss += iter_loss.item()
            samples = Variable(torch.Tensor(x_test))
            labels = y_test
            if use_gpu:
                samples = samples.cuda()
            preds = tree_net(samples)
            if use_gpu:
                preds = preds.cpu()
            if regression:
                acc = mean_squared_error(labels, preds.detach().numpy())
            else:
                preds = torch.argmax(preds, dim=1)
                correct = labels[labels == preds.numpy()]
                acc = round(len(correct)/float(len(y_test)), 2)
            if epoch % log_every == 0:
                print(f"[Epoch: {epoch} loss: {epoch_train_loss}  acc: {acc}]")
        tree_fn = 'tree_'+str(tree_number)+'.pth.tar'
        save_checkpoint(tree_net, filename=tree_fn)
        print("Model " + tree_fn + " saved")
        tree_number += 1
        tree_list.append(tree_net)

    return tree_list


def continue_training_trees(tree_list, x_in, y_in, lr=1e-3, num_epochs=10, batch_size=1, regression=False, log_every=5):
    """
    take in existing models and keep training
    :param tree_list: as returned from above
    :param x_in: input data
    :param y_in: labels
    :param lr: learning rate
    :param epochs: epochs to train
    :param batch_size: batch size
    :return: tree_list again but more trained
    """
    tree_list_to_return = []
    for tree_number, tree_net in enumerate(tree_list):

        opt = torch.optim.Adam(tree_net.parameters(), lr=lr)

        if regression:
            loss = nn.MSELoss()
        else:
            loss = nn.CrossEntropyLoss()

        if use_gpu:
            loss = loss.cuda()
            tree_net = tree_net.cuda()
        for epoch in range(num_epochs):
            x_train, x_test, y_train, y_test = train_test_split(x_in, y_in, test_size=0.1)

            epoch_train_loss = 0
            for ind in range(0, len(x_train), batch_size):
                samples = Variable(torch.Tensor(x_train[ind:ind+batch_size]))
                if regression:
                    labels = Variable(torch.Tensor(y_train[ind:ind + batch_size]))
                else:
                    labels = Variable(torch.LongTensor(y_train[ind:ind+batch_size]))
                if use_gpu:
                    samples = samples.cuda()
                    labels = labels.cuda()
                preds = tree_net(samples)
                if regression:
                    preds = preds.view(-1)
                iter_loss = loss(preds, labels)
                opt.zero_grad()
                iter_loss.backward()
                opt.step()
                epoch_train_loss += iter_loss.item()
            samples = Variable(torch.Tensor(x_test))
            labels = y_test
            if use_gpu:
                samples = samples.cuda()
            preds = tree_net(samples)
            if use_gpu:
                preds = preds.cpu()
            if regression:
                acc = mean_squared_error(labels, preds.detach().numpy())
            else:
                preds = torch.argmax(preds, dim=1)
                correct = labels[labels == preds.numpy()]
                acc = round(len(correct) / float(len(y_test)), 2)
            if epoch % log_every == 0:
                print(f"[Epoch: {epoch} loss: {epoch_train_loss}  acc: {acc}]")
        tree_fn = 'tree_'+str(tree_number)+'.pth.tar'
        save_checkpoint(tree_net, filename=tree_fn)
        print("Model " + tree_fn + " saved")
        tree_number += 1
        tree_list_to_return.append(tree_net)

    return tree_list_to_return

