import random
from scipy.stats import pearsonr,spearmanr
from scipy.spatial.distance import euclidean
import math

def randomHex():
    s = '#'
    for i in range(6):
        s+=random.choice(['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'])
    return s

def spearmanDistance(u,v):
    if len(set(u)) == 1 and len(set(v)) == 1 and u[0] == v[0]: #Prevent NaN values
        return 0
    elif len(set(u)) == 1 and len(set(v)) == 1 and u[0] != v[0]:
        return 1
    elif len(set(u)) == 1 or len(set(v)) == 1:
        return euclidean(u,v)/math.sqrt(len(v)) #normalized euclidean distance
    return 1 - spearmanr(u,v)[0]

def pearsonDistance(u,v):
    if len(set(u)) == 1 and len(set(v)) == 1 and u[0] == v[0]:  # Prevent NaN values
        return 0
    elif len(set(u)) == 1 and len(set(v)) == 1 and u[0] != v[0]:
        return 1
    elif len(set(u)) == 1 or len(set(v)) == 1:
        return euclidean(u, v) / math.sqrt(len(v))  # normalized euclidean distance
    return 1 - pearsonr(u,v)[0]