"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # TODO: score how mixed the labels are; 0 for a pure set, larger for more mixed sets.
    if(labels.size == 0):
        return 0.0

    unique_values,counts = np.unique(labels,return_counts = True)
    return 1 - np.sum((counts/np.sum(counts))**2)

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # Create boolean masks directly
    left_mask = features[:, feature_index] <= threshold
    right_mask = features[:, feature_index] > threshold

    lx = features[left_mask]
    rx = features[right_mask]
    ly = labels[left_mask]
    ry = labels[right_mask]

    return lx, ly, rx, ry

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    return impurity(parent_labels) - (left_labels.size*impurity(left_labels)+right_labels.size*impurity(right_labels))/parent_labels.size

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    best_feature_index = None
    best_threshold = None
    best_score = 0 # Safer initialization for maximization

    # Loop through the passed feature_indices instead of all columns
    for i in feature_indices:
        vals = np.unique(features[:, i])
        
        # Skip if there are fewer than 2 unique values to split on
        if len(vals) < 2:
            continue
            
        thresholds = (vals[:-1] + vals[1:]) / 2
        
        for j in thresholds:
            lx, ly, rx, ry = split_dataset(features, labels, i, j)
            score = split_score(labels, ly, ry)
            
            if score > best_score:
                best_feature_index = i
                best_threshold = j
                best_score = score
    
    return {'feature_index':best_feature_index,'threshold':best_threshold,"score":best_score}

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # TODO: decide whether to stop growing based on purity, depth, and size...
    if(np.unique(labels).size == 1 or depth>=max_depth or labels.shape[0]<min_samples_split):
        return True
    return False

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

