import math

from numpy import *

from utils.measurement import root_mean_square
from utils.tools import convert2matrix, mdotl, feature_scaling
from utils.plot import draw_gd_debug

def least_square(features, values, w, lmd=None, regularizer=2):
    """featuresn least square method impletement

    Use sum of square error function which aroses in the maximum likehood
    to get liner regression param set w for supervised learning.
    To control over-fitting, use lmd for regularization.

    input:
    features - inputs of the training data set, represented as a list.
            eg. [(x11,x12...x1n),(x21,x22...x2n),...]
    values - outputs of training data set, represented as a list.
            eg. [(y10,y11...y1m),(y20,y21...y2m),...]
    lmd, regularizer - regularization parameter

    output:
    param set w
    """
    if lmd == None:
        lmd = 0
    size = len(values)

    R = lmd * eye(len(features[0, :]))
    # solve the equation features'*features*w=features'b
    w = linalg.solve(R+dot(features.T, features), dot(features.T, values))
    return w


def gradient_descent(features, values, w,
                     sigma=None, steps=None, scaling=True, debug=False):
    if sigma == None:
        sigma = 1e-2
    if steps == None:
        steps = 10000
    debug_y = []
    debug_x = list(range(0, steps, math.floor((steps/100))))

    size = len(values)

    if scaling:
        features = feature_scaling(features)
    for i in range(steps):
        # (features*w - b) * features is the derivative term
        if debug and i in debug_x:
            debug_y.append(root_mean_square(features, values, w))
        w = w - sigma * dot((dot(features, w) - values), features) / size
    if debug:
        draw_gd_debug(
                    debug_x, debug_y,
                    'sigma:{}_steps:{}_scaling:{}'.format(sigma, steps, scaling)
        )
    return w


def bayesian(features, values, w, alpha=None, beta=None, steps=None):
    """
    linear Regression use bayesian approach
    """
    #alpha and beta can be init by user
    if alpha == None:
        alpha = .1
    if beta == None:
        beta = 10
    #iterate time control to fix alpha beta
    if steps == None:
        steps = 100
    size = len(values)
    R = alpha * eye(len(features[0, :]))
    SD = eye(len(features[0, :])) / alpha

    #calculate eig values for lmd in order to fix alpha, beta
    eig, v = linalg.eig(dot(features.T, features))
    for step in range(steps):
        SD = linalg.inv(R + beta * dot(features.T, features))
        w = beta * mdotl(SD, features.T, values)
        gamma = ((beta * eig) / (alpha + beta * eig)).sum()
        alpha = gamma / dot(w, w.T)
        beta = (size - gamma) / ((values - dot(features, w)) ** 2).sum()

    return w, SD, alpha, beta
