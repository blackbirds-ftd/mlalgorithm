import mlalgorithm.pltset as pltset
import math
from numpy import *
import scipy
import copy
import matplotlib.pyplot as plt


def leastsqu(dataSet, targetSet, regParam = 0, regularizer = 2):
	"""
	**************************************************************************************
	Use sum of square error function which aroses in the maximum likehood to get liner 
	regression param set w for supervised learning. To control over-fitting, use regParam 
	for	regularization

	input: dataSet -- Training dataSet input, eg. [[x11,x12...x1n][x21,x22...x2n]...]
				 targetSet -- Training dataSet Output, eg. [[y10,y11...y1m][y20,y21...y2m]...],
											targetSet and dataSet should have same length
				 regParam, regularizer -- regularization parameter

	output: param set w
					#voice precisor value b
	**************************************************************************************
	"""
	#X = copy.deepcopy(dataSet)
	#for data in X:
	#	data.insert(0,1)
	X = array(dataSet)
	Y = array(targetSet)
	
	try:
		len(dataSet) == len(targetSet)
	except:
		print 'dataSet and targetSet should have the same length'
	
	reg = 0.5*regParam*regularizer*eye(len(X[0]))
	w = linalg.inv(reg + dot(transpose(X), X))
	w = dot(w, transpose(X))
	w = dot(w, Y)
	
	T = Y.sum(axis = 0)/len(Y)
	Phi = X.sum(axis = 0)/len(X)
	w0 = T - dot(Phi, transpose(w)[0])

	w = transpose(w)[0].tolist()
	w.insert(0, w0.tolist()[0])
		
	pltset.drawset(dataSet, targetSet, w)
	return w



def bayeslinear(dataSet, targetSet):
	"""
	linear Regression use bayesian approach
	"""
	X = copy.deepcopy(dataSet)
	for data in X:
		data.insert(0,1)
	X = array(X)
	Y = array(targetSet)
	
	try:
		len(X) == len(Y)
	except:
		print 'dataSet and targetSet should have the same length'

	priorPrecision = 11.1
	likehoodSD = 0.002
	likehoodPrecision = 1/(likehoodSD)**2

	priorMean = zeros((len(X[0]), 1))
	priorDev = eye(len(X[0]))/priorPrecision
	
	postMean = priorMean
	postDev = priorDev

	for i in xrange(len(dataSet)):
		data = X[i].reshape(1, len(X[i]))
		target = Y[i].reshape(1, len(Y[i]))

		postDev = linalg.inv(linalg.inv(priorDev)+likehoodPrecision*dot(transpose(data), data))
		postMean = dot(postDev, dot(linalg.inv(priorDev), priorMean)+likehoodPrecision*dot(transpose(data), target))

		#if i%2 == 0:
		#	filename = r'bayes' + str(i)
		#	drawset(dataSet, targetSet, postMean, filename)
		priorMean = postMean
		priorDev = postDev
	
	pltset.drawset(dataSet, targetSet, transpose(postMean)[0].tolist())
	return transpose(postMean)[0].tolist(), postDev.tolist() 


