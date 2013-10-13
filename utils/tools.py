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
