import numpy as np

def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    # Your code here
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    P = np.sum(y_true==1)
    N = np.sum(y_true==0)
    if P==0 or N==0:
        return 0.0
    
    ord = np.argsort(-y_scores)
    y_scores = y_scores[ord]
    y_true = y_true[ord]

    tp = np.cumsum(y_true==1)
    fp = np.cumsum(y_true==0)

    indices = np.where(np.diff(y_scores))[0]
    indices = np.append(indices,len(y_scores)-1)

    tpr = np.append(0,tp[indices]/P)
    fpr = np.append(0,fp[indices]/N)

    return np.sum((tpr[1:]+tpr[:-1])/2 * (fpr[1:]-fpr[:-1]))

    # return np.trapezoid(tpr,fpr)


