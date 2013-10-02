#!/usr/bin/python3
# -*- coding: utf-8 -*-
from utils.data import generate_2dcurve_dots, sin2pix, build_polynomial_function
from utils.plot import draw_curve
from utils.measurement import root_mean_square
from linear.regression import least_square
import os


def fit_2dcurve(xset, yset, method=least_square, basis='poly', M=9, lmd=0):
    """
    """
    features = [tuple(x**i for i in range(1, M+1)) for x in xset]
    w = method(features, yset, lmd)
    if not os.path.exists('result/'):
        os.mkdir('result')
    draw_curve(sin2pix, xset, yset, w,
               '_'.join(['result/ls',
                         'size{}'.format(len(xset)),
                         'M{}'.format(M),
                         '{}_ regularization'.format('with' if lmd else 'no')
                     ]))
    return w


def print_result(training_data, test_data, M=9, lmd=0.0):
    print('M={}, lmd={:.10}'.format(M, lmd))
    w = fit_2dcurve(training_data[0], training_data[1], M=M, lmd=lmd)
    f = build_polynomial_function(w)
    print('RMS for training set: {}\nfor test set: {}'.format(
        root_mean_square(training_data[1], list(map(f, training_data[0]))),
        root_mean_square(test_data[1], list(map(f, test_data[0])))))

if __name__ == '__main__':
    print('#####some curve fitting examples.#####')
    training10 = generate_2dcurve_dots(10)
    training50 = generate_2dcurve_dots(50)
    test50 = generate_2dcurve_dots(50)
    print('use 10 dots for training')
    print_result(training10, test50, M=1)
    print_result(training10, test50, M=3)
    print_result(training10, test50, M=9)
    print_result(training10, test50, M=9, lmd=.001)
    print('use 50 dots for training')
    print_result(training50, test50, M=9)




