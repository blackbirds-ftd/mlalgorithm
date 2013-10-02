from numpy import *

def root_mean_square(s, d):
    assert len(s) == len(d)
    return sqrt(((array(s) - array(d))**2).sum() / len(s))
