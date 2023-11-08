import numpy as np

def scale_score(score, pdo = 60, base_odd = 1, base_score = 600):
    alpha = pdo/np.log(2)
    scaled_score = base_score - alpha*np.log(base_odd) + alpha*np.log((1-score)/score)
    return scaled_score