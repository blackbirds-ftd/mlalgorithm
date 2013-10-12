from utils.tools import mdotl
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


