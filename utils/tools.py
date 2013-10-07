from numpy import *

def convert2matrix(features, values, w=None):
        length = min(len(features), len(values))
        A = array(features[:length])
        b = array(values[:length])
        if w == None:
            return A, b, length
        w = array(w)
        return A, b, length, w
