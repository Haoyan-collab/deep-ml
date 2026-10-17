import torch

def bagging_classifier(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, n_estimators: int = 10, seed: int = 42) -> torch.Tensor:
    """
    Implement a bagging classifier using decision stumps.
    
    Args:
        X_train: Training features of shape (n_samples, n_features), torch.float32
        y_train: Training labels of shape (n_samples,), binary {0, 1}, torch.long
        X_test: Test features of shape (n_test_samples, n_features), torch.float32
        n_estimators: Number of bootstrap samples/base estimators
        seed: Random seed for reproducibility
    
    Returns:
        torch.Tensor: Predicted labels for X_test as a 1D long tensor
    """
    # Your code here
    rng = torch.Generator(device=X_train.device).manual_seed(seed)
    n_samples, n_features = X_train.shape

    all_pred = []
    for _ in range(n_estimators):
        best_error = float('inf')
        best_threshold = -1
        best_feat_idx = -1
        indices = torch.randint(
            0, n_samples, (n_samples,),
            generator=rng,
            device=X_train.device
        )
        x_sample = X_train[indices]
        y_sample = y_train[indices]
        for j in range(n_features):
            feat = x_sample[:,j]
            for threshold in feat:
                pred = torch.where(feat>threshold, 1, 0)
                error = torch.sum(pred!=y_sample, dim=-1)
                if error < best_error:
                    best_error = error
                    best_threshold = threshold
                    best_feat_idx = j
        feat = X_test[:,best_feat_idx]
        pred = torch.where(feat>best_threshold, 1, 0)
        all_pred.append(pred)

    all_pred = torch.stack(all_pred) 
    votes = torch.sum(all_pred, dim=0)
    votes = torch.where(votes>n_estimators/2, 1, 0)
    return votes











