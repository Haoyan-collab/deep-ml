import torch

def calculate_auc(y_true, y_scores) -> float:
    """
    Calculate the Area Under the ROC Curve (AUC) using PyTorch.
    
    Args:
        y_true: List, array, or torch.Tensor of binary ground truth labels (0 or 1)
        y_scores: List, array, or torch.Tensor of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    # Your code here
    y_scores = torch.tensor(y_scores)
    y_true = torch.tensor(y_true)
    P = torch.sum(y_true==1)
    N = torch.sum(y_true==0)
    if P == 0 or N == 0:
        return 0.0
    
    ord = torch.argsort(-y_scores)
    y_scores = y_scores[ord]
    y_true = y_true[ord]
    tp = torch.cumsum(y_true==1,dim=0)
    fp = torch.cumsum(y_true==0,dim=0)

    indices = torch.where(torch.diff(y_scores))[0]
    indices = torch.cat([indices,torch.tensor([len(y_scores)-1])])
    tpr = torch.cat([torch.zeros(1),tp[indices]/P])
    fpr = torch.cat([torch.zeros(1),fp[indices]/N])

    result = torch.sum((tpr[:-1]+tpr[1:])/2 * (fpr[1:]-fpr[:-1]))
    return result.item()