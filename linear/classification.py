from numpy import *
from utils.data import sigfunc
from utils.tools import convert2matrix, mdotl
from utils.plot import draw_gd_debug
from utils.measurement import root_mean_square


def perceptron(features, labels, w, steps=None):
    """
    The most defficult problem for perceptron is to judge whether the sample is linear seperable. In an other word, I dont know if the sample is not linear seperable or just learn too slow to get the convergence.
    """
    if steps == None:
        steps = 200
    size = len(labels)
    step = 0

    # judge this sample is not linear seperable after repeat 5 times.
    for rep in range(steps):
        for i in range(size):
            if dot(w, features[i])*labels[i] >= 0:
                continue
            step += 1
            w = w + features[i] * labels[i]

        # check if all the dataset are correctly classified
        if array([(dot(w, features.T)*labels)[k]>0
                    for k in range(size)]).all() == True:
            print('Reach Linear Seperable!')
            break
        if rep == steps - 1:
            print('Cannot Reach Linear Seperable after training data set for {} times!'.format(steps))
    return w, step


def gradient_descent(features, labels, w, sigma=None, steps=None):
    """
    """
    if sigma == None:
        sigma = 1e-2
    if steps == None:
        steps = 10000
    debug_x = list(range(0, steps, math.floor((steps/100))))
    debug_y = []
    size = len(labels)

    for step in range(steps):
        if step in debug_x:
            debug_y.append(root_mean_square(features, labels, w, sigfunc))
        w = w - sigma * dot((sigfunc(dot(features, w)) - labels), features) / size

    draw_gd_debug(debug_x, debug_y, 'temp')

    return w


def IRLS(features, labels, w, steps=None):
    """
    Namely replace learning rate with Hessian Matrix. According to root_mean_square evaluation, this algorithm is depend on lucky? Odd!
    """
    if steps == None:
        steps = 100
    debug_x = list(range(0, steps, math.floor((steps/100))))
    debug_y = []
    size = len(labels)

    R = eye(size)
    for step in range(steps):
        if step in debug_x:
            debug_y.append(root_mean_square(features, labels, w, sigfunc))

        Y = sigfunc(dot(features, w))
        for i in range(size):
            R[i][i] = Y[i] * (1 - Y[i])
        z = dot(features, w) - dot(linalg.pinv(R),(Y - labels))
        w = mdotl(linalg.pinv(mdotl(features.T, R, features)), features.T, R, z)

    draw_gd_debug(debug_x, debug_y, 'temp')
    return w
