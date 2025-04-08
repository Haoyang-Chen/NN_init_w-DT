import torch
import torch.nn as nn
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import torch
import torch.nn as nn
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def initialize_nn_from_tree(tree_clf, input_dim):
    tree = tree_clf.tree_
    max_depth = tree.max_depth
    node_to_row_idx = {}  # (node_id, polarity) -> row index used across layers
    row_counter = 0
    paths = []

    # Traverse tree and collect full paths including leaf nodes
    def traverse(node_id, path):
        nonlocal row_counter
        if tree.feature[node_id] == -2:  # Leaf node
            path_with_leaf = path + [(node_id, None)]  # Leaf has no polarity
            paths.append(path_with_leaf)
            return

        # TRUE (left) child
        node_to_row_idx[(node_id, True)] = row_counter
        row_counter += 1
        traverse(tree.children_left[node_id], path + [(node_id, True)])

        # FALSE (right) child
        node_to_row_idx[(node_id, False)] = row_counter
        row_counter += 1
        traverse(tree.children_right[node_id], path + [(node_id, False)])

    traverse(0, [])

    num_logic_units = len(node_to_row_idx)
    num_outputs = tree.value.shape[1]

    weights = []
    biases = []

    # --- Layer 0: Input to split logic layer (Sigmoid) ---
    w0 = torch.zeros((num_logic_units, input_dim))
    b0 = torch.zeros(num_logic_units)

    for (node_id, polarity), row_idx in node_to_row_idx.items():
        feature = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        w0[row_idx, feature] = 1 if polarity else -1
        b0[row_idx] = -threshold if polarity else threshold

    weights.append(w0)
    biases.append(b0)

    # --- Layers 1 to max_depth: Logic construction layers (ReLU) ---
    for l in range(max_depth):
        w = torch.zeros((num_logic_units, num_logic_units))
        b = torch.zeros(num_logic_units)

        for path in paths:
            if len(path) < l + 2:
                continue  # No parent-child pair at this layer

            parent, child = path[l], path[l+1]
            if parent[1] is None or child[1] is None:
                continue  # Skip invalid (leaf) path segment

            parent_idx = node_to_row_idx[parent]
            child_idx = node_to_row_idx[child]
            w[child_idx, parent_idx] = 1 if child[1] else -1

            # Łukasiewicz AND logic bias
            if parent[1] and child[1]:
                b[child_idx] = -1
            elif parent[1] != child[1]:
                b[child_idx] = 0
            else:  # both False
                b[child_idx] = 1

        weights.append(w)
        biases.append(b)

    # --- Pass-through Layers: For shorter paths, propagate to last layer ---
    final_layer = max_depth
    for path in paths:
        if len(path) < final_layer + 1:
            last_node = path[-1]
            if last_node[1] is None:
                continue
            start_idx = node_to_row_idx[last_node]
            for layer in range(len(path)-1, final_layer):
                w = torch.zeros((num_logic_units, num_logic_units))
                b = torch.zeros(num_logic_units)
                w[start_idx, start_idx] = 1.0  # Identity connection
                weights.append(w)
                biases.append(b)

    # --- Output Layer: Connect leaf nodes to their classes ---
    w_out = torch.zeros((num_outputs, num_logic_units))
    b_out = torch.zeros(num_outputs)

    for path in paths:
        leaf_node = path[-1]
        if tree.feature[leaf_node[0]] != -2:
            continue  # Sanity check
        leaf_idx = node_to_row_idx.get(leaf_node)
        if leaf_idx is None:
            continue
        class_counts = tree.value[leaf_node[0]][0]
        for cls in range(num_outputs):
            if class_counts[cls] > 0:
                w_out[cls, leaf_idx] = 1.0  # Exact connection

    weights.append(w_out)
    biases.append(b_out)

    # --- Assemble Network ---
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
            layers.append(linear)  # Output: no activation

    model = nn.Sequential(*layers)
    return model



from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

nn_model = initialize_nn_from_tree(clf, input_dim=X.shape[1], output_dim=len(np.unique(y)))
print(nn_model)
