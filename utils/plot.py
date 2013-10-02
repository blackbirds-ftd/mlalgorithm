import math
from numpy import arange
import matplotlib.pyplot as plt

def draw_curve(f, xset, yset, w=None, filename=None):
        """
        Plot baisi Function(green line),
        Regression Function(red line),
        Training data(blue points, default zeros)
        """
        step = arange(0, 1, .01)
        regression = [
                x if x**2 < 4 else 0
                for x in (sum([w[i]*x**i for i in range(len(w))]) for x in step)
        ]
        basis = list(map(f, step))

        plt.figure()
        ax = plt.subplot(111)
        ax.scatter(xset, yset, c = 'b')
        ax.plot(step, basis, c = 'g')
        ax.plot(step, regression, c = 'r')
        if filename:
                plt.savefig(filename)
        plt.show()

