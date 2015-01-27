# -*- coding: utf-8 -*-

from sklearn.datasets import load_boston
from sklearn.preprocessing import normalize

from mhml.learner import OnlineGradientDescentLearner
from mhml.model import FactorMachine, FieldFactorMachine, Param, Inst


def doregression(learner, X, y):
    for i in xrange(20000):
        print 'round:\t%d' % (i)
        cost = 0
        for i in xrange(len(y)):
            if isinstance(learner.model, FactorMachine):
                inst = Inst(zip(xrange(len(X[i])), X[i]), y[i])
            elif isinstance(learner.model, FieldFactorMachine):
                numfeatures = len(X[i])
                inst = Inst(zip(zip(xrange(numfeatures), [1]*numfeatures), X[i]), y[i])
            p = learner.predict(inst)
            cost += (p - inst.y)**2
            learner.update(inst)
        print 'total cost:', cost,
        print 'average cost', cost/len(y),
        print


def testFM():
    boston = load_boston()
    X = normalize(boston.data, axis=0)
    y = boston.target

    X = X[:len(X)/10]
    y = y[:len(y)/10]

    param = Param(K=2)
    fmlearner = OnlineGradientDescentLearner(FactorMachine(param), param)
    ffmlearner = OnlineGradientDescentLearner(FieldFactorMachine(param), param)

    print 'Factor Machine'
    doregression(fmlearner, X, y)
    print 'Field Factor Machine'
    doregression(ffmlearner, X, y)

if __name__ == '__main__':
    test_FM()
