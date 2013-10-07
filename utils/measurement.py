from math import sqrt
from numpy import *
from utils.tools import convert2matrix

def root_mean_square(features, values, w):
    A, b, size, w = convert2matrix(features, values, w)
    v = dot(A, w) - b
    return sqrt(dot(v, v) / size)
