import random
import math
from numpy import *

def dataGen():
        """
        *********************************************************************
        From basis function gen dataSet and targetSet with Gaussian noise
        *********************************************************************
        """
        randSet = []
        resultSet = []
        noiseSet = []
        for i in range(20): #No, of points in a Set
                randnum = random.uniform(0, 1)
                randSet.append(randnum)
                resultSet.append(sin(2*math.pi*randnum)) #Gen function
                #resultSet.append(0.5*randnum-0.3)
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

