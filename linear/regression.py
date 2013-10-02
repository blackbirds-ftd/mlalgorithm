from numpy import *

def least_square(features, values, lmd=0, regularizer=2):
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
        #voice precisor value b
        """
        length = min(len(features), len(values))
        X = array(features[:length])
        Y = array(values[:length])

        R = lmd*eye(len(X[0, :]))
        # solve the equation X'*X*w=X'Y
        w = linalg.solve(R+dot(X.T, X), dot(X.T, Y))
        w0 = Y.sum() / len(Y) - dot(X.sum(axis=0) / len(X[:, 0]), w)

        w = w.tolist()
        w[0:0] = [w0]
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
                print( 'dataSet and targetSet should have the same length')

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
                #       filename = r'bayes' + str(i)
                #       drawset(dataSet, targetSet, postMean, filename)
                priorMean = postMean
                priorDev = postDev
        
#       pltset.drawset(dataSet, targetSet, transpose(postMean)[0].tolist())
        return transpose(postMean)[0].tolist(), postDev.tolist() 


