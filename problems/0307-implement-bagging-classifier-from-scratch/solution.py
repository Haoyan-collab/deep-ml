import numpy as np

def bagging_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_estimators: int = 10, seed: int = 42) -> np.ndarray:
    """
    Implement a bagging classifier using decision stumps.
    
    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,), binary {0, 1}
        X_test: Test features of shape (n_test_samples, n_features)
        n_estimators: Number of bootstrap samples/base estimators
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Predicted labels for X_test
    """
    # Your code here
    rng = np.random.default_rng(seed)

    n_samples, n_features = X_train.shape
    all_pred = []
    for _ in range(n_estimators):
        indices = rng.choice(n_samples,size = n_samples,replace=True)
        X_sample = X_train[indices]
        Y_sample = y_train[indices]

        best_error = np.inf
        best_feature = -1
        best_threshold = -1

        for j in range(n_features):
            values = np.unique(X_sample[:,j])
            for val in values:
                threshold = val
                pred = np.where(X_sample[:,j] > threshold, 1, 0)
                error = np.sum(pred!=Y_sample, axis=-1)

                if error < best_error:
                    best_error = error
                    best_feature = j
                    best_threshold = val

        pred = np.where(
            X_test[:,best_feature] > best_threshold, 1, 0)
        all_pred.append(pred)

    all_pred = np.array(all_pred)
    votes = np.sum(all_pred, axis=0) # votes on each samples from every tree
    final_pred = np.where(votes > n_estimators/2, 1, 0)
    return final_pred











