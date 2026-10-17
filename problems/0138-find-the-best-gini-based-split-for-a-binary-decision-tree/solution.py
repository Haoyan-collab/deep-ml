import torch
from typing import Tuple

def find_best_split(X: torch.Tensor, y: torch.Tensor) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    # ✏️ TODO: implement
    def gini(labels):
        if len(labels) == 0:
            return 0.0
        p = torch.mean(labels.float(), dim=-1)
        return 1-p**2-(1-p)**2
    
    idx = -1
    best_threshold = -1
    best_gini = float('inf')
    m = X.shape[0]
    n = X.shape[1]
    for j in range(n):
        feat = X[:,j]
        uni_feat = torch.unique(feat)
        for threshold in uni_feat:
            left_mask = feat <= threshold
            left_pred = y[left_mask]
            right_pred = y[~left_mask]

            left_gini = gini(left_pred)
            right_gini = gini(right_pred)
            weighted_gini = (len(left_pred)/m)*left_gini + (len(right_pred)/m)*right_gini

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                idx = j
                best_threshold = threshold

    return (idx,best_threshold.item())
