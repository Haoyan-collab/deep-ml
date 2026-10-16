import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    # ✏️ TODO: implement
    def gini(labels):
        if len(labels) == 0:
            return 0.0
        p = np.mean(labels, axis=-1)
        return 1-p**2-(1-p)**2
    n, d = X.shape
    # for each feature, iterate the sequence
    idx, mingini = -1, np.inf
    best_thresh = -1
    for j in range(d):
        values = np.unique(X[:,j])
        # list all threshold for this feature
        for i in range(len(values)-1):
            threshold = values[i]
            left_mask = X[:,j] <= threshold  # [True,True,False,False]
            # y[left_mask] : [0,1,0,1] -> [0,1]
            left_y = y[left_mask]
            right_y = y[~left_mask]

            left_gini = gini(left_y)
            right_gini = gini(right_y)
            weighted_gini = (len(left_y)/n)*left_gini + (len(right_y)/n)*right_gini
            if weighted_gini < mingini:
                mingini = weighted_gini
                best_thresh = threshold
                idx = j

    return (idx,best_thresh)