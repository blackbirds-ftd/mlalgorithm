import random
import math
from numpy import *
import scipy
import copy
import matplotlib.pyplot as plt

def dataGen():
	"""
	**************************************************************************************
	From f(n) = sin(2*pi*x) gen dataSet and targetSet with Gaussian noise
	**************************************************************************************
	"""
	randSet = []
	resultSet = []
	noiseSet = []
	for i in range(9): #No, of points in a Set
		randnum = random.uniform(0, 1)
		randSet.append(randnum)
		resultSet.append(math.sin(2*math.pi*randnum)) #Gen function
		noiseSet.append(random.normal(0, 0.2))
	dataSet = []
	targetSet = []
	for i in range(len(randSet)):
		data = []
		target = [resultSet[i]+noiseSet[i]]

		#base function, here is x^j
		for j in range(1,10):
			data.append(randSet[i]**j)
		dataSet.append(data)
		targetSet.append(target)
		
	return dataSet, targetSet



def linearreg(dataSet, targetSet, regParam = 0, regularizer = 2):
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
	X = copy.deepcopy(dataSet)
	for data in X:
		data.insert(0,1)
	X = array(X)
	Y = array(targetSet)
	
	try:
		len(X) == len(Y)
	except:
		print 'dataSet and targetSet should have the same length'
	
	reg = 0.5*regParam*regularizer*ones((len(X[0]), len(X[0])))
	w = linalg.inv(reg + dot(transpose(X), X))
	w = dot(w, transpose(X))
	w = dot(w, Y)
	
	#T = Y.sum(axis = 0)/len(Y)
	#O = X.sum(axis = 0)/len(X)
	#w[0] = 0
	#w[0] = T - dot(O, transpose(w)[0])

	retList = []
	for i in range(len(w)):
		retList.append(array(w)[i].tolist()[0])
	
	return retList



def drawset(dataSet, targetSet, w = []):
	"""
	**************************************************************************************
	Plot Gen Function(green line), Regression Function(red line), Training points(blue points, default zeros)
	**************************************************************************************
	"""
	if len(w) == 0:
		w = [0 for i in range(len(dataSet[0]))]

	randSet = []
	nSet = []
	for i in xrange(len(dataSet)):
		randSet.append(dataSet[i][0])
		nSet.append(targetSet[i][0])
	
	x = arange(0, 1, 0.01)
	sinfunc = []
	regressionfunc = []
	for i in x:
		reg = 0
		for j in range(len(w)):
			reg += w[j]*i**j
		regressionfunc.append(reg)
		sinfunc.append(math.sin(2*math.pi*i))
	
	plt.figure()
	ax = plt.subplot(111)
	ax.scatter(randSet, nSet, c = 'b')
	ax.plot(x, sinfunc, c = 'g')
	ax.plot(x, regressionfunc, c = 'r')
	plt.show()
	#return randSet, nSet, x, sinfunc, regressionfunc

	
