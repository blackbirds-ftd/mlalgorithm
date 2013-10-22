#!/usr/bin/python3
# -*- coding: utf-8 -*-
from utils.data import generate_2d2label_dots, linear
from utils.plot import draw_class
from utils.tools import convert2matrix, changelabel
from linear.classification import perceptron, gradient_descent, IRLS
import os

if not os.path.exists('result/'):
    os.mkdir('result')


def classification(features, labels, w, method, **kwargs):
    """
    """
    if method.__name__ == 'perceptron':
        w, step = method(features, labels, w, **kwargs)
    elif method.__name__ == 'gradient_descent':
        w = method(features, labels, w, **kwargs)
    elif method.__name__ == 'IRLS':
        w = method(features, labels, w, **kwargs)

    draw_picture(features, labels, w, method.__name__)
    return w


def draw_picture(features, labels, w, method, **kwargs):
    features = features[:, 1:] #descard bias features
    draw_class(linear, features, labels, w)


def use_perceptron(training_features, training_labels, steps=None):
    m = len(training_features[0]) + 1
    training_features, training_labels, w = convert2matrix(
                                            training_features, training_labels,
                                            [1 for i in range(m)],
                                            addbias=True)
    w = classification(
        training_features, training_labels, w, perceptron,
        steps=steps)


def use_gradient_descent(training_features, training_labels,
                         sigma=None, steps=None):
    m = len(training_features[0]) + 1
    training_features, training_labels, w = convert2matrix(
                                            training_features, training_labels,
                                            [1 for i in range(m)],
                                            addbias=True)
    w = classification(
        training_features, training_labels, w, gradient_descent,
        sigma=sigma, steps=steps)

def use_IRLS(training_features, training_labels, steps=None):
    m = len(training_features[0]) + 1
    training_features, training_labels, w = convert2matrix(
                                            training_features, training_labels,
                                            [1 for i in range(m)],
                                            addbias=True)
    w = classification(
        training_features, training_labels, w, IRLS,
        steps=steps)


if __name__ == '__main__':
    print('#####some classification examples.#####')
    train10X, train10Y = generate_2d2label_dots(10)
    train20X, train20Y = generate_2d2label_dots(20)
    train50X, train50Y = generate_2d2label_dots(50)
    train100X, train100Y = generate_2d2label_dots(100)
    print('#####use perceptron.#####')
    use_perceptron(train10X, train10Y)
    use_perceptron(train20X, train20Y)
    use_perceptron(train50X, train50Y)
    use_perceptron(train100X, train100Y)
    # change label coding type from (-1, 1) to (0, 1)
    train10Y = changelabel(train10Y)
    train20Y = changelabel(train20Y)
    train50Y = changelabel(train50Y)
    train100Y = changelabel(train100Y)
    print("#####use gradient descent.#####")
    use_gradient_descent(train10X, train10Y)
    use_gradient_descent(train20X, train20Y)
    use_gradient_descent(train50X, train50Y)
    use_gradient_descent(train100X, train100Y)
    print('#####use iterative reweighted least squares.#####')
    use_IRLS(train10X, train10Y)
    use_IRLS(train20X, train20Y)
    use_IRLS(train50X, train50Y)
    use_IRLS(train100X, train100Y)

