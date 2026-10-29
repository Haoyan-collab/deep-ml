import numpy as np

def top_p_sampling(logits: list[float], p: float) -> list[float]:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a list of floats
    """
    n = len(logits)
    logits = np.array(logits)
    logits -= np.max(logits, axis=-1)
    logits = np.exp(logits)/np.sum(np.exp(logits), axis=-1)

    indices = sorted(range(n), key=lambda i: (-logits[i],i))
    nuclues = []
    sm = 0
    for i in indices:
        nuclues.append(i)
        sm += logits[i]
        if sm >= p:
            break
    result = [0] *n
    for i in nuclues:
        result[i] = logits[i]

    result = result/np.sum(result, axis=-1)
    return np.round(result, 4).tolist()