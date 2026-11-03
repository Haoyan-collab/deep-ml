import numpy as np

def compute_confusion_matrix(y_true, y_pred, num_classes, normalize=None, round_decimals=4):
    """
    Compute a KxK confusion matrix with optional normalization.

    Args:
        y_true: Iterable of true labels in [0, K-1]
        y_pred: Iterable of predicted labels in [0, K-1]
        num_classes: K, number of classes
        normalize: None | 'true' | 'pred' | 'all'
        round_decimals: decimals to round when normalization is applied

    Returns:
        list[list[int|float]] confusion matrix
    """
    # Your implementation here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = np.zeros((num_classes,num_classes))
    np.add.at(cm,(y_true,y_pred),1)

    if normalize == 'true':
        denom = np.sum(cm,axis = -1,keepdims = True)
        cm = np.divide(cm,denom,out=np.zeros_like(cm),where=denom != 0)

    elif normalize == 'pred':
        denom = np.sum(cm,axis = 0,keepdims = True)
        cm = np.divide(cm,denom,out=np.zeros_like(cm),where=denom != 0)

    elif normalize == 'all':
        denom = cm.sum()
        if denom>0:
            cm /= denom

    return np.round(cm,decimals = round_decimals).tolist()
