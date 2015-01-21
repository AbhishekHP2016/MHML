# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris

from mhml.learner import OnlineGradientDescentLearner
from mhml.model import LogisticRegression, Param, Inst
from mhml.util import logloss


def test_LR():
    iris = load_iris()
    X = iris.data
    y = [1.0 if target >= 1. else 0. for target in iris.target]

    learner = OnlineGradientDescentLearner(LogisticRegression(), Param())

    for j in xrange(1000):
        print 'round:\t%d' % j
        cost = 0
        for i in xrange(len(y)):
            inst = Inst(zip(xrange(len(X[i])), X[i]), y[i])
            p = learner.predict(inst)
            inst_cost = logloss(p, inst.y)
            cost += inst_cost
            learner.update(inst)
        print 'total cost:', cost
        print 'average cost', cost/len(y)

if __name__ == '__main__':
    test_LR()
