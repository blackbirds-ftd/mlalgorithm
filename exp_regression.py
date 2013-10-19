#!/usr/bin/python3
# -*- coding: utf-8 -*-
from utils.data import generate_2dcurve_dots, sin2pix
from utils.plot import draw_curve, draw_gauss
from utils.measurement import root_mean_square
from utils.tools import feature_scaling
from linear.regression import least_square, gradient_descent, bayesian
import os

if not os.path.exists('result/'):
    os.mkdir('result')
if not os.path.exists('debug/'):
    os.mkdir('debug')


least_square_string = 'M:{M}, lmd:{lmd:.10}'
gradient_descent_string = 'M:{M}, sigma:{sigma}, steps:{steps}, scaling:{scaling}'
bayesian_string = 'M:{M}, init alpha:{alpha}, init beta:{beta}, steps:{steps}'

def generate_polynomial_features(r, feature):
    """Return a tuple of features using polynomial model

    arguments:
    r - range, exp. (0,2) represent features in (x**0, x**1, x**2)
    feature - orignal feature
    """
    return tuple(feature**i for i in range(r[0], r[1]+1))


def convert2polynomial(r, feature_set):
    """Convert every feature in feature set to polynomial model"""
    return [generate_polynomial_features(r, x) for x in feature_set]


def generate_filename(method, **kwargs):
    """Return result picture name with the given param"""
    least_square = '{method}_size:{size}_M:{M}_regularize:{lmd}'
    gradient_descent = '{method}_size:{size}_M:{M}_scaling:{scaling}'
    bayesian = '{method}_size:{size}_M:{M}'

    output = ''
    if method == 'least_square':
        output = least_square
    elif method == 'gradient_descent':
        output = gradient_descent
    elif method == 'bayesian':
        output = bayesian
    return output.format(method=method, **kwargs).replace('.', '-')


def print_RMS_result(training, test, w, scaling=False):
    training_features = convert2polynomial((0, len(w)-1), training[0])
    test_features = convert2polynomial((0, len(w)-1), test[0])
    if scaling:
        training_features = feature_scaling(training_features)
        test_features = feature_scaling(test_features)
    print('RMS for training set: {}\nfor test set: {}'.format(
        root_mean_square(training_features, training[1], w),
        root_mean_square(test_features, test[1], w),
    ))


def fit_2dcurve(xset, yset, method, **kwargs):
    """
    """
    M = kwargs.pop('M', None)
    w = kwargs.pop('w', None)

    if method.__name__ == 'least_square':
        features = convert2polynomial((1, M), xset)
        w = method(features, yset, **kwargs)
    elif method.__name__ == 'gradient_descent':
        features = convert2polynomial((0, M), xset)
        w = method(features, yset, w, **kwargs)
    elif method.__name__ == 'bayesian':
        features = convert2polynomial((0, M), xset)
        w, SD, alpha, beta = method(features, yset, w, **kwargs)
        print('post alpha={}, post beta={}'.format(alpha,beta))

    kwargs.update({'M': M, 'size': len(xset)})
    filename = generate_filename(method.__name__, **kwargs)
    if method.__name__ == 'bayesian':
        draw_picture(xset, yset, w, method.__name__, filename, SD=SD, beta=beta)
    else:
        draw_picture(xset, yset, w, method.__name__, filename)
    return w


def draw_picture(xset, yset, w, method, filename, **kwargs):
    if method == 'bayesian':
        SD = kwargs.get('SD')
        beta = kwargs.get('beta')
        draw_gauss(sin2pix, xset, yset, w, SD, beta, 'result/'+filename)
    else:
        draw_curve(sin2pix, xset, yset, w, 'result/'+filename)


def use_least_square(training_data, test_data, M=9, lmd=0.0):
    print(least_square_string.format(M=M, lmd=lmd))

    w = fit_2dcurve(training_data[0], training_data[1],
                    least_square, M=M, lmd=lmd)

    print_RMS_result(training_data, test_data, w)


def use_gradient_descent(training_data, test_data, M=9,
                         sigma=None, steps=None, scaling=True, debug=False):
    print(gradient_descent_string.format(
        M=M, scaling=scaling, sigma=sigma or 1e-2, steps=steps or 10000
    ))

    w = fit_2dcurve(
        training_data[0], training_data[1] ,gradient_descent,
        M=M, w=[0 for i in range(M + 1)], sigma=sigma,
        steps=steps, scaling=scaling, debug=debug
    )

    print_RMS_result(training_data, test_data, w, scaling)


def use_bayesian(training_data, test_data, M=9,
                 alpha=None, beta=None, steps=None):
    print(bayesian_string.format(
        M=M, alpha=alpha or 0.1, beta=beta or 10, steps=steps or 1
    ))

    w = fit_2dcurve(
        training_data[0], training_data[1], bayesian,
        M=M, w=[0 for i in range(M + 1)], alpha=alpha, beta=beta, steps=steps
    )

    print_RMS_result(training_data, test_data, w)


if __name__ == '__main__':
    print('#####some curve fitting examples.#####')
    training1 = generate_2dcurve_dots(1)
    training2 = generate_2dcurve_dots(2)
    training10 = generate_2dcurve_dots(10)
    training50 = generate_2dcurve_dots(50)
    test50 = generate_2dcurve_dots(50)
    print('#####use lease square.#####')
    print('use 10 dots for training')
    use_least_square(training10, test50, M=1)
    use_least_square(training10, test50, M=3)
    use_least_square(training10, test50, M=9)
    use_least_square(training10, test50, M=9, lmd=.001)
    print('use 50 dots for training')
    use_least_square(training50, test50, M=9)
    print('#####use gradient descent.#####')
    print('use 50 dots for training')
    use_gradient_descent(training50, test50, M=9, sigma=0.5, steps=10000, scaling=False, debug=True)
    use_gradient_descent(training50, test50, M=9, sigma=2, steps=10000, debug=True)
    print('#####use bayesian regression.#####')
    print('use 1 dots for training')
    use_bayesian(training1, test50, M=9)
    print('use 2 dots for training')
    use_bayesian(training2, test50, M=9)
    print('use 10 dots for training')
    use_bayesian(training10, test50, M=9)
    print('use 50 dots for training')
    use_bayesian(training50, test50, M=9, alpha=.01, beta=100, steps=5)
    use_bayesian(training50, test50, M=9, alpha=.01, beta=100, steps=10)
    use_bayesian(training50, test50, M=9, alpha=.01, beta=100, steps=15)
    print('see pic in result dir')
