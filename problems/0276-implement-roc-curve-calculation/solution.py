import numpy as np

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    # Your code here
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    P = np.sum(y_true==1)
    N = np.sum(y_true==0)
    if P==0 or N==0:
        return None
    
    ord = np.argsort(-y_scores)
    y_scores = y_scores[ord]
    y_true = y_true[ord]
    tp = np.cumsum(y_true==1)
    fp = np.cumsum(y_true==0)

    indices = np.where(np.diff(y_scores))[0]
    indices = np.append(indices,len(y_scores)-1)
    tpr = np.append([0],tp[indices]/P)
    fpr = np.append([0],fp[indices]/N)
    return(fpr.tolist(),tpr.tolist())
