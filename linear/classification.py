from numpy import *
from utils.tools import convert2matrix


def perceptron(features, labels, w):
	"""
	The most defficult problem for perceptron is to judge whether the sample is linear seperable. In an other word, I dont know if the sample is not linear seperable or just learn too slow to get the convergence.
	"""
	A, t, size, w = convert2matrix(features, labels, w)
	steps = 0
	# judge this sample is not linear seperable after repeat 5 times.
	for rep in range(200):
		for i in range(size):
			if dot(w.T, A[i])*t[i] >= 0:
				continue
			steps += 1
			w = w + A[i] * t[i]
		# check if all the dataset are correctly classified
		if array([(dot(w.T, A.T)*t)[k]>0 for k in range(size)]).all() == True:
			print('Reach Linear Seperable!')
			break
		else:
			if rep == 199:
				print('Cannot Reach Linear Seperable after training data set for 200 times!')
	return w, steps


