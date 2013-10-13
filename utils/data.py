import random
import functools

from  math import sin, pi


# def build_polynomial_function(coefficient):
#         """return a polynomial function with given coefficient"""
#         def f(x):
#                 coe = coefficient
#                 return sum([coe[i]*x**i for i in range(len(coe))])
#         return f

def sin2pix(x):
        return sin(2*pi*x)

def linear(x):
	return 5 - x

def step_func(a):
	if a>= 0:
		return 1
	else:
		return -1

def noise(f):
        """Add Gaussian noise to observed values of data set"""
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
                features, values = f(*args, **kwargs)
                return features, [v + random.gauss(0, 0.2) for v in values]
        return wrapper

@noise
def generate_2dcurve_dots(N=20, f=sin2pix, r=(0, 1)):
        """Generate a dot set from f of size N in range r"""
        xset = [(random.uniform(*r)) for i in range(N)]
        yset = list(map(f, xset))
        return xset, yset

def label(f):
	"""Add label 1 if above or equal to zero, -1 if below zero"""
	@functools.wraps(f)
	def wrapper(*args, **kwargs):
		features, labels = f(*args, **kwargs)
		return features, list(map(step_func, labels))
	return wrapper

@label
def generate_2d2label_dots(N=20, f=linear, r=(0, 5)):
	"""Generate dot with 2 labels"""
	x = [(random.uniform(*r)) for i in range(N)]
	y = [(random.uniform(*r)) for i in range(N)]
	xset = [[x[i], y[i]] for i in range(N)]
	yset = [list(map(f, x))[i] - y[i] for i in range(N)]
	return xset, yset 

