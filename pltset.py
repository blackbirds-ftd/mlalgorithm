import math
import matplotlib.pyplot as plt
from numpy import *

def drawset(dataSet, targetSet, w = [], filename = ''):
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
        basisfunc = []
        regressionfunc = []
        for i in x:
                reg = 0
                for j in range(len(w)):
                        reg += w[j]*i**j
                if(reg**2>=4):
                        regressionfunc.append(0)
                else:
                        regressionfunc.append(reg)
                basisfunc.append(math.sin(2*math.pi*i))
                #basisfunc.append(0.5*i-0.3)

        plt.figure()
        ax = plt.subplot(111)
        ax.scatter(randSet, nSet, c = 'b')
        ax.plot(x, basisfunc, c = 'g')
        ax.plot(x, regressionfunc, c = 'r')
        if filename != '':
                plt.savefig(filename + 'png')
        plt.show()

