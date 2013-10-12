#!/usr/bin/python3
# -*- coding: utf-8 -*-
from utils.data import generate_2dcurve_dots, sin2pix
from utils.plot import draw_curve, draw_gauss
from utils.measurement import root_mean_square
from linear.regression import least_square, gradient_descent, bayesian
import os


def fit_2dcurve(xset, yset, method, basis='poly', **kwargs):
    """
    """
    M = kwargs.get('M')
    if method.__name__ == 'least_square':
        features = [tuple(x**i for i in range(1, M+1)) for x in xset]
        w = method(features, yset, kwargs.get('lmd'))
    elif method.__name__ == 'gradient_descent':
        features = [tuple(x**i for i in range(0, M+1)) for x in xset]
        w = [0 for i in range(M + 1)]
        w = method(features, yset, w,
                   kwargs.get('sigma'), kwargs.get('steps'))
    elif method.__name__ == 'bayesian':
<<<<<<< HEAD
        features = [tuple(x**i for i in range(0, M+1)) for x in xset]
	w = [0 for i in range(M + 1)]
        w, SD, alpha, beta = method(features, yset, w,
                                    kwargs.get('alpha'), 
                                    kwargs.get('beta'), 
                                    kwargs.get('steps'))
	#for convenient use
=======
        features = [tuple(x**i for i in range(1, M+1)) for x in xset]
        w, sd, alpha, beta = method(features, yset,
		   kwargs.get('alpha'), 
		   kwargs.get('beta'), 
		   kwargs.get('step'))
		#should return post alpha and beta
>>>>>>> 1492e9462aa6257c49d93ec850e8967a0674bce9
        print('post alpha={}, post beta={}'.format(alpha,beta))

    if method.__name__ == 'bayesian':
	draw_picture(xset, yset, w, method.__name__, SD=SD, beta=beta)
    else:
	draw_picture(xset, yset, w, method.__name__)    
    return w


def draw_picture(xset, yset, w, method, **kwargs):
    if not os.path.exists('result/'):
        os.mkdir('result')
    file_name = ''
    if method == 'least_square':
        file_name = '_'.join(['result/ls',
                              'size{}'.format(len(xset)),
                              'M{}'.format(kwargs.get('M')),
                              '{}_regularization'.format(
                                  'with' if kwargs.get('lmd') else 'no')
                          ])
    elif method == 'gradient_descent':
        file_name = '_'.join(['result/gd',
                              'size{}'.format(len(xset)),
                              'M{}'.format(kwargs.get('M')),
                          ])
    elif method == 'bayesian':
        file_name = '_'.join(['result/bayes',
<<<<<<< HEAD
                              'size{}'.format(len(xset)),
                              'M{}'.format(kwargs.get('M')),
                          ])
	SD = kwargs.get('SD')
	beta = kwargs.get('beta')
    
    if method == 'bayesian':
        draw_gauss(sin2pix, xset, yset, w, SD, beta, file_name)
    else:
        draw_curve(sin2pix, xset, yset, w, file_name)
=======
			      'size{}'.format(len(xset)),
			      'M{}'.format(kwargs.get('M')),
			  ])

    draw_curve(sin2pix, xset, yset, w, file_name)
>>>>>>> 1492e9462aa6257c49d93ec850e8967a0674bce9


def use_least_square(training_data, test_data, M=9, lmd=0.0):
    print('M={}, lmd={:.10}'.format(M, lmd))
    w = fit_2dcurve(training_data[0], training_data[1],
                    least_square, M=M, lmd=lmd)
    print('RMS for training set: {}\nfor test set: {}'.format(
        root_mean_square(
            [tuple(x**i for i in range(0, M+1)) for x in training_data[0]],
            training_data[1],
            w),
        root_mean_square(
            [tuple(x**i for i in range(0, M+1)) for x in test_data[0]],
            test_data[1],
            w
        )))


def use_gradient_descent(training_data, test_data, M=9, sigma=None, steps=None):
    print('M={}, sigma={}, steps={}'.format(M,
                                            sigma if sigma else 1e-2,
                                            steps if steps else 10000))
    w = fit_2dcurve(training_data[0], training_data[1],
                    gradient_descent, M=M, sigma=sigma, steps=steps)
    print('RMS for training set: {}\nfor test set: {}'.format(
        root_mean_square(
            [tuple(x**i for i in range(0, M+1)) for x in training_data[0]],
            training_data[1],
            w),
        root_mean_square(
            [tuple(x**i for i in range(0, M+1)) for x in test_data[0]],
            test_data[1],
            w
        )))

def use_bayesian(training_data, test_data, M=9, alpha=None, beta=None, steps=None):
    print('M={}, init alpha={}, init beta={}, steps={}'.format(M,
<<<<<<< HEAD
                                                               alpha if alpha else 0.1,
                                                               beta if beta else 10,
                                                               steps if steps else 100))
=======
			     				       alpha if alpha else 0.1,
							       beta if beta else 10,
							       steps if steps else 100))
>>>>>>> 1492e9462aa6257c49d93ec850e8967a0674bce9
    w = fit_2dcurve(training_data[0], training_data[1],
					bayesian, M=M, alpha=alpha, beta=beta, steps=steps)
    print('RMS for training set: {}\nfor test set: {}'.format(
        root_mean_square(
	    [tuple(x**i for i in range(0, M+1)) for x in training_data[0]],
	    training_data[1],
	    w),
        root_mean_square(
	    [tuple(x**i for i in range(0, M+1)) for x in test_data[0]],
	    test_data[1],
	    w)
        ))

if __name__ == '__main__':
    print('#####some curve fitting examples.#####')
    print('#####use lease square.#####')
    training10 = generate_2dcurve_dots(10)
    training50 = generate_2dcurve_dots(50)
    test50 = generate_2dcurve_dots(50)
    print('use 10 dots for training')
    use_least_square(training10, test50, M=1)
    use_least_square(training10, test50, M=3)
    use_least_square(training10, test50, M=9)
    use_least_square(training10, test50, M=9, lmd=.001)
    print('use 50 dots for training')
    use_least_square(training50, test50, M=9)
    print('#####use gradient descent.#####')
    print('use 50 dots for training')
    use_gradient_descent(training50, test50, M=3)
    use_gradient_descent(training50, test50, M=9, sigma=.005, steps=100000) 
    print('#####use bayesian regression.#####')
    print('use 10 dots for training')
    use_bayesian(training10, test50, M=9)
    print('use 50 dots for training')
    use_bayesian(training50, test50, M=9, alpha=.1, beta=25, steps=1)
    use_bayesian(training50, test50, M=9, alpha=.1, beta=25, steps=2)
    use_bayesian(training50, test50, M=9, alpha=.1, beta=25, steps=3)
    use_bayesian(training50, test50, M=9, alpha=.1, beta=25, steps=4)
    print('see pic in result dir')

