from math import sqrt
from numpy import *
from utils.tools import convert2matrix

def root_mean_square(features, values, w):

    size = len(features[0]) + 1
    v = dot(features, w) - values
    return sqrt(dot(v, v) / size)
