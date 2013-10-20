import functools
import operator
from numpy import *

def convert2matrix(features, values, w=None, addbias=False):
        length = min(len(features), len(values))
        A = array(features[:length])
        b = array(values[:length])
        if addbias:
            A = addones(A)
        if w == None:
            return A, b
        w = array(w)
        return A, b, w


def addones(X):
    return c_[ones((len(X), 1)), X]


def feature_scaling(A, skip_first_col=True):
        """Feature scaling"""
        if not isinstance(A, ndarray):
                try:
                        A = array(A)
                except:
                        return A

        if not skip_first_col:
                return mean_normalize(A)
        B = A[:, 1:]
        return column_stack([A[:,:1], mean_normalize(B)])

def mean_normalize(A):
        return ((A - outer(ones(A.shape[0]) ,A.mean(axis=0)))
                / outer(ones(A.shape[0]), (A.max(axis=0) - A.min(axis=0))))

def mdotl(*args):
        return functools.reduce(dot, args)
#def mdot(*args):
#should be dot(a,dot(dot(b,c),d)) == mdot(a, ((b, c), d))

# brutial use
def seperate_from_label(features, labels):
        label_type = set(labels)
        x = [[],[]]
        for i in range(len(features)):
                if labels[i] == 1:
                        x[0].append(features[i])
                if labels[i] == -1:
                        x[1].append(features[i])
        return x, len(label_type)
