import torch
import torch.nn as nn
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def initialize_nn_from_tree(tree_clf, input_dim):
    tree = tree_clf.tree_
    max_depth = tree.max_depth
    node_to_neuron = {}  # (node_id, polarity) -> neuron_id
    neuron_counter = 0
    paths = []

    # Traverse the tree to collect paths and assign neurons
    def traverse(node_id, path):
        nonlocal neuron_counter
        if tree.feature[node_id] == -2:  # Leaf node
            paths.append(path[:])
            return

        # TRUE branch (left child)
        node_to_neuron[(node_id, True)] = neuron_counter
        neuron_counter += 1
        traverse(tree.children_left[node_id], path + [(node_id, True)])

        # FALSE branch (right child), skip root's false
        if node_id != 0:
            node_to_neuron[(node_id, False)] = neuron_counter
            neuron_counter += 1
        traverse(tree.children_right[node_id], path + [(node_id, False)])

    traverse(0, [])

    layer_size = len(node_to_neuron)
    num_outputs = tree.value.shape[1]

    weights = []
    biases = []

    # --- Layer L=0: Input to Threshold Units (Sigmoid) ---
    w0 = torch.zeros((layer_size, input_dim))
    b0 = torch.zeros(layer_size)

    for (node_id, polarity), neuron_idx in node_to_neuron.items():
        feature = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        w0[neuron_idx, feature] = 1 if polarity else -1
        b0[neuron_idx] = -threshold if polarity else threshold

    weights.append(w0)
    biases.append(b0)

    # --- Layers L=1 to L=max_depth: Logical Construction (ReLU) ---
    for l in range(max_depth):
        w = torch.zeros((layer_size, layer_size))
        b = torch.zeros(layer_size)

        for path in paths:
            if len(path) < l + 2:
                continue
            parent, child = path[l], path[l+1]
            if parent not in node_to_neuron or child not in node_to_neuron:
                continue
            parent_idx = node_to_neuron[parent]
            child_idx = node_to_neuron[child]

            # Set weight based on child's polarity
            w[child_idx, parent_idx] = 1 if child[1] else -1

            # Set bias using Łukasiewicz AND logic
            if parent[1] and child[1]:
                b[child_idx] = -1
            elif parent[1] != child[1]:
                b[child_idx] = 0
            else:
                b[child_idx] = 1

        weights.append(w)
        biases.append(b)

    # --- Output Layer: OR of Leaves ---
    w_out = torch.zeros((num_outputs, layer_size))
    b_out = torch.zeros(num_outputs)

    for path in paths:
        leaf_node_id = path[-1][0]
        majority_class = int(np.argmax(tree.value[leaf_node_id]))
        leaf_neuron = node_to_neuron.get(path[-1])
        if leaf_neuron is not None:
            w_out[majority_class, leaf_neuron] = 1  # OR logic

    weights.append(w_out)
    biases.append(b_out)

    # --- Build Full Model ---
    layers = []

    for i, (w, b) in enumerate(zip(weights, biases)):
        linear = nn.Linear(w.shape[1], w.shape[0])
        with torch.no_grad():
            linear.weight.copy_(w)
            linear.bias.copy_(b)

        if i == 0:
            layers.append(nn.Sequential(linear, nn.Sigmoid()))
        elif i < len(weights) - 1:
            layers.append(nn.Sequential(linear, nn.ReLU()))
        else:
            layers.append(linear)  # Output layer: no activation

    model = nn.Sequential(*layers)
    return model


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

nn_model = initialize_nn_from_tree(clf, input_dim=X.shape[1])
print(nn_model)
