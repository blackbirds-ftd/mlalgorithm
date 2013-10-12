from numpy import *
from utils.measurement import root_mean_square
from utils.tools import convert2matrix, mdotl

def least_square(features, values, lmd=None, regularizer=2):
        """An one-value least square method impletement

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
        A, b, size = convert2matrix(features, values)

        R = lmd * eye(len(A[0, :]))
        # solve the equation A'*A*w=A'b
        w = linalg.solve(R+dot(A.T, A), dot(A.T, b))
        w0 = (b - dot(A, w)).sum() / size

        w = w.tolist()
        w[0:0] = [w0]
        return w


def gradient_descent(features, values, w, sigma=None, steps=None):
        if sigma == None:
                sigma = 1e-2
        if steps == None:
                steps = 10000
        A, b, size, w = convert2matrix(features, values, w)

        for i in range(steps):
                # (A*w - b) * A is the derivative term
                w = w - sigma * dot((dot(A, w) - b), A)
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
		steps = 10
	postalpha = alpha
	postbeta = beta

	A, b, size, Mean = convert2matrix(features, values, w)
	R = alpha * eye(len(A[0, :]))
	SD = eye(len(A[0, :])) / alpha

	#calculate eig values for lmd in order to fix alpha, beta
	eig, v = linalg.eig(dot(A.T, A))	
	for step in range(steps):
		alpha = postalpha
		beta = postbeta

		SD = linalg.inv(R + beta * dot(A.T, A))
		Mean = beta * mdotl(SD, A.T, b)

		gamma = ((beta * eig) / (alpha + beta * eig)).sum()
		postalpha = gamma / dot(Mean, Mean.T)
		postbeta = (size - gamma) / ((b - dot(A, Mean)) ** 2).sum()

		if alpha == postalpha and beta == postbeta:
			break

        return Mean, SD, alpha, beta


