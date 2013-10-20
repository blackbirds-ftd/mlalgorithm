#!/usr/bin/python3
# -*- coding: utf-8 -*-
from utils.data import generate_2dcurve_dots, sin2pix
from utils.plot import draw_curve, draw_gauss
from utils.measurement import root_mean_square
from utils.tools import feature_scaling, convert2matrix
from linear.regression import least_square, gradient_descent, bayesian
import os

if not os.path.exists('result/'):
    os.mkdir('result')
if not os.path.exists('debug/'):
    os.mkdir('debug')


least_square_string = 'M:9, lmd:{lmd:.10}'
gradient_descent_string = 'M:9, sigma:{sigma}, steps:{steps}, scaling:{scaling}'
bayesian_string = 'M:9, init alpha:{alpha}, init beta:{beta}, steps:{steps}'

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
    least_square = '{method}_size:{size}_M:9_regularize:{lmd}'
    gradient_descent = '{method}_size:{size}_M:9_scaling:{scaling}'
    bayesian = '{method}_size:{size}_M:9'

    output = ''
    if method == 'least_square':
        output = least_square
    elif method == 'gradient_descent':
        output = gradient_descent
    elif method == 'bayesian':
        output = bayesian
    return output.format(method=method, **kwargs).replace('.', '-')


def print_RMS_result(training_features, training_values,
                     test_features, test_values,
                     w, scaling=False):
    if scaling:
        training_features = feature_scaling(training_features)
        test_features = feature_scaling(test_features)
    print('RMS for training set: {}\nfor test set: {}'.format(
        root_mean_square(training_features, training_values, w),
        root_mean_square(test_features, test_values, w),
    ))


def regression(features, values, w, method, **kwargs):
    """
    """
    if method.__name__ == 'least_square':
        w = method(features, values, w, **kwargs)
    elif method.__name__ == 'gradient_descent':
        w = method(features, values, w, **kwargs)
    elif method.__name__ == 'bayesian':
        w, SD, alpha, beta = method(features, values, w, **kwargs)
        print('post alpha={}, post beta={}'.format(alpha,beta))

    kwargs.update({'size': len(features)})
    filename = generate_filename(method.__name__, **kwargs)
    if method.__name__ == 'bayesian':
        draw_picture(features, values, w, method.__name__, filename,
                     SD=SD, beta=beta)
    else:
        draw_picture(features, values, w, method.__name__, filename)
    return w


def draw_picture(xset, yset, w, method, filename, **kwargs):
    if method == 'bayesian':
        SD = kwargs.get('SD')
        beta = kwargs.get('beta')
        draw_gauss(sin2pix, xset, yset, w, SD, beta, 'result/'+filename)
    else:
        draw_curve(sin2pix, xset, yset, w, 'result/'+filename)


def use_least_square(training_features, training_values,
                     test_features, test_values,
                     lmd=0.0):
    print(least_square_string.format(lmd=lmd))
    # m represent feature numbers
    m = len(training_features[0]) + 1
    training_features, training_values, w = convert2matrix(
                                            training_features, training_values,
                                            [0 for i in range(m)],
                                            addbias=True)
    test_features, test_values = convert2matrix(
                                 test_features, test_values, addbias=True)
    w = regression(
            training_features, training_values, w, least_square,
            lmd=lmd)

    print_RMS_result(training_features, training_values,
                     test_features, test_values, w)


def use_gradient_descent(training_features, training_values,
                         test_features, test_values,
                         sigma=None, steps=None, scaling=True, debug=False):
    print(gradient_descent_string.format(
        scaling=scaling, sigma=sigma or 1e-2, steps=steps or 10000
    ))
    # m represent feature numbers
    m = len(training_features[0]) + 1
    training_features, training_values, w = convert2matrix(
                                            training_features, training_values,
                                            [0 for i in range(m)],
                                            addbias=True)
    test_features, test_values = convert2matrix(
                                 test_features, test_values, addbias=True)
    w = regression(
        training_features, training_values, w, gradient_descent,
        sigma=sigma,
        steps=steps, scaling=scaling, debug=debug
    )

    print_RMS_result(training_features, training_values,
                     test_features, test_values, w)


def use_bayesian(training_features, training_values,
                 test_features, test_values,
                 alpha=None, beta=None, steps=None):
    print(bayesian_string.format(
        alpha=alpha or 0.1, beta=beta or 10, steps=steps or 1
    ))
    # m represent feature numbers
    m = len(training_features[0]) + 1
    training_features, training_values, w = convert2matrix(
                                            training_features, training_values,
                                            [0 for i in range(m)],
                                            addbias=True)
    test_features, test_values = convert2matrix(
                                 test_features, test_values, addbias=True)
    w = regression(
        training_features, training_values, w, bayesian,
        alpha=alpha, beta=beta, steps=steps
    )

    print_RMS_result(training_features, training_values,
                     test_features, test_values, w)


if __name__ == '__main__':
    print('#####some curve fitting examples.#####')
    #generate original dataset
    train2X, train2Y = generate_2dcurve_dots(2)
    train10X, train10Y = generate_2dcurve_dots(10)
    train50X, train50Y = generate_2dcurve_dots(50)
    test50X, test50Y = generate_2dcurve_dots(50)
    #feature pre-treatment: convert to polynomial features
    train2X = convert2polynomial((1, 9), train2X)
    train10X = convert2polynomial((1, 9), train10X)
    train50X = convert2polynomial((1, 9), train50X)
    test50X = convert2polynomial((1, 9), test50X)
    """print('#####use lease square.#####')
    print('use 10 dots for training')
    use_least_square(train10X, train10Y, test50X, test50Y)
    use_least_square(train10X, train10Y, test50X, test50Y, lmd=.001)
    print('use 50 dots for training')
    use_least_square(train50X, train50Y, test50X, test50Y)
    print('#####use gradient descent.#####')
    print('use 50 dots for training')
    use_gradient_descent(train50X, train50Y, test50X, test50Y,
                         sigma=0.5, steps=10000, scaling=False, debug=True)
    use_gradient_descent(train50X, train50Y, test50X, test50Y,
                         sigma=2, steps=10000, debug=True)"""
    print('#####use bayesian regression.#####')
    print('use 2 dots for training')
    use_bayesian(train2X, train2Y, test50X, test50Y)
    print('use 10 dots for training')
    use_bayesian(train10X, train10Y, test50X, test50Y)
    print('use 50 dots for training')
    use_bayesian(train50X, train50Y, test50X, test50Y,
                 alpha=.01, beta=100, steps=5)
    use_bayesian(train50X, train50Y, test50X, test50Y,
                 alpha=.01, beta=100, steps=10)
    use_bayesian(train50X, train50Y, test50X, test50Y,
                 alpha=.01, beta=100, steps=15)
    print('see pic in result dir')
