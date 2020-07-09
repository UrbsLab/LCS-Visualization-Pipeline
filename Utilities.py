from scipy.stats import pearsonr,spearmanr
from scipy.spatial.distance import euclidean
import math
import numpy as np

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

def find_elbow(data_list): #data is a list of y axis numbers. Assume x is 1 to len(data)
    '''
    Inspired by: https://datascience.stackexchange.com/questions/57122/in-elbow-curve-how-to-find-the-point-from-where-the-curve-starts-to-rise
    '''

    data = np.array(list(zip(range(1,len(data_list)+1),data_list)))
    theta = get_data_theta(data)

    # make rotation matrix
    co = np.cos(theta)
    si = np.sin(theta)
    rotation_matrix = np.array(((co, si), (-si, co)))

    # rotate data vector
    rotated_vector = data.dot(rotation_matrix)

    # return x value of elbow
    return np.where(rotated_vector[:,1] == rotated_vector[:,1].min())[0][0] + 1 #+1 transforms index to actual x value

def get_data_theta(data): #in radians
    return np.arctan2(abs(data[:, 1].max() - data[:, 1].min()), abs(data[:, 0].max() - data[:, 0].min()))