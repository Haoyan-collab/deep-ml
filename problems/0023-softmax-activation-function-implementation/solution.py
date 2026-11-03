import math
import numpy as np
def softmax(scores: list[float]) -> list[float]:
    scores = np.array(scores)
    scores -= np.max(scores,axis = -1)
    return np.exp(scores)/np.sum(np.exp(scores)).tolist()