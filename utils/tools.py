from numpy import *

def convert2matrix(features, values, w=None):
        length = min(len(features), len(values))
        A = array(features[:length])
        b = array(values[:length])
        if w == None:
            return A, b, length
        w = array(w)
        return A, b, length, w


def mdotl(*args):
	return reduce(dot, args)

#def mdot(*args):
#should be dot(a,dot(dot(b,c),d)) == mdot(a, ((b, c), d))
