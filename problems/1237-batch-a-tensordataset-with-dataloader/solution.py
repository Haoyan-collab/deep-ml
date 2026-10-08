import torch
from torch.utils.data import TensorDataset, DataLoader

def batch_stats(X, y):
    """Wrap X and y in TensorDataset + DataLoader(batch_size=4, shuffle=False).

    Return (num_batches, first_batch_X_shape_tuple).
    """
    # TODO
    dataset = TensorDataset(X,y) #X.shape[0] must equal to y.shape[0]
    loader = DataLoader(dataset,batch_size = 4, shuffle = False)

    cnt = 0
    first_shape = None

    for b_x, b_y in loader:
        cnt += 1
        if cnt == 1:
            first_shape = tuple(b_x.shape)
    
    return (cnt, first_shape)
