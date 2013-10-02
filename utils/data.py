import random
import functools

from  math import sin, pi


def build_polynomial_function(coefficient):
        """return a polynomial function with given coefficient"""
        def f(x):
                coe = coefficient
                return sum([coe[i]*x**i for i in range(len(coe))])
        return f

def sin2pix(x):
        return sin(2*pi*x)


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

