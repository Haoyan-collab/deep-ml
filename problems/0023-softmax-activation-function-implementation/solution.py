import math
import numpy as np
def softmax(scores: list[float]) -> list[float]:
    max_scores = max(scores)
    exp_scores = [np.exp(x - max_scores) for x in scores]
    prob = exp_scores / sum(exp_scores)
    return prob