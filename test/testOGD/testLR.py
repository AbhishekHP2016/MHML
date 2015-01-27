# -*- coding: utf-8 -*-

'''This file test the validity of Logistic Regression and
Generalized Logisitc Regression.

The testing is based on the data in scikit-learn.
'''

from sklearn.datasets import load_iris

from mhml.learner import OnlineGradientDescentLearner
from mhml.model import (
    LogisticRegression,
    GeneralizedLogisticRegression,
    FactorMachine,
    FieldFactorMachine,
    Param,
    Inst,
    ismodel
)

from mhml.util import logloss


def doregression(learner, X, y):
    for j in xrange(5000):
        cost = 0
        for i in xrange(len(y)):
            if ismodel(learner.model, FieldFactorMachine):
                index = zip(xrange(len(X[i])), (1 for i in xrange(len(X[i]))))
                inst = Inst(zip(index, X[i]), y[i])
            else:
                inst = Inst(zip(xrange(len(X[i])), X[i]), y[i])
            p = learner.predict(inst)
            inst_cost = logloss(p, inst.y)
            cost += inst_cost
            learner.update(inst)
        if not j % 1000:
            print 'round:', j
            print 'total cost:', cost
            print 'average cost', cost/len(y)


def testLR():
    iris = load_iris()
    X = iris.data
    y = [1.0 if target >= 1. else 0. for target in iris.target]

    param = Param(L2=0.01, D=20, K=2)
    lrlearner = OnlineGradientDescentLearner(LogisticRegression(param.D), param)
    fmglrlearner = OnlineGradientDescentLearner(
        GeneralizedLogisticRegression(FactorMachine(param)), param)
    ffmglrlearner = OnlineGradientDescentLearner(
        GeneralizedLogisticRegression(FieldFactorMachine(param)), param)
    # print 'Logistic Regression:'
    # doregression(lrlearner, X, y)
    # print 'Logistic Regression(Factor Machine):'
    # doregression(fmglrlearner, X, y)
    print 'Logistic Regression(Field Factor Machine):'
    doregression(ffmglrlearner, X, y)

if __name__ == '__main__':
    testLR()
