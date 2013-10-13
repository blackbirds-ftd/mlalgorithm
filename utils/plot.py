from utils.tools import mdotl, seperate_from_label
from scipy.stats import norm
from numpy import *
import math
import matplotlib.pyplot as plt

def draw_curve(f, xset, yset, w=None, filename=None):
        """
        Plot baisi Function(green line),
        Regression Function(red line),
        Training data(blue points, default zeros)
        """
        step = linspace(0, 1, 100)
        regression = [
                x 
                for x in (sum([w[i]*x**i for i in range(len(w))]) for x in step)
        ]
        basis = list(map(f, step))

        plt.figure()
        plt.subplot(111)
        plt.xlim(0, 1)
        plt.ylim(-1.5, 1.5)
        plt.scatter(xset, yset, color='b')
        plt.plot(step, basis, color='g')
        plt.plot(step, regression, color='r')

        if filename:
                plt.savefig(filename)
        plt.show()

def draw_gauss(f, xset, yset, w, SD, beta, filename=None):
	def cal_std(x):
		psi = array([w[i]*x**i for i in range(len(w))])
		std = (1/beta + mdotl(psi.T, SD, psi))**0.5
		return std

        step = linspace(0, 1, 100)
        regression = [
                x
                for x in (sum([w[i]*x**i for i in range(len(w))]) for x in step)
        ]
	std = list(map(cal_std, step))
        basis = list(map(f, step))

	Y1 = array([norm(regression[i], std[i]).interval(0.95)[0] for i in range(len(step))])
	Y2 = array([norm(regression[i], std[i]).interval(0.95)[1] for i in range(len(step))])

	plt.figure()
        plt.subplot(111)
        plt.xlim(0, 1)
        plt.ylim(-1.5, 1.5)
	plt.fill_between(step, Y1, Y2, color='pink')
	plt.scatter(xset, yset, color='b')
        plt.plot(step, basis, color='g')
	plt.plot(step, regression, color='r')
	
        if filename:
                plt.savefig(filename)
        plt.show()

def draw_class(f, xset, yset, w=None, filename=None):
	"""blue for label 1
	   yellow for label 2
	"""
        step = linspace(0, 5, 500)
        boundary = [
                x 
                for x in (sum([w[i]*x**i for i in range(len(w)-1)])/-w[len(w)-1]
		for x in step)
        ]
        basis = list(map(f, step))
	x, type_num = seperate_from_label(xset, yset)

        plt.figure()
        plt.subplot(111)
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        plt.scatter(array([x[0][i][0] for i in range(len(x[0]))]), 
		    array([x[0][i][1] for i in range(len(x[0]))]), 
		    color='b')
        plt.scatter(array([x[1][i][0] for i in range(len(x[1]))]), 
		    array([x[1][i][1] for i in range(len(x[1]))]), 
		    color='y')
        #plt.plot(step, basis, color='g')
        plt.plot(step, boundary, color='r')

        if filename:
                plt.savefig(filename)
        plt.show()
