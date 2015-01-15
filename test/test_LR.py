# coding: utf-8
import sys

from sklearn.datasets import load_iris

sys.path.append('../')

from learner import OgdLearner
from model import LR, Param, Inst
from commonfun import logloss


if __name__ == '__main__':

    iris = load_iris()
    X = iris.data
    y = [1.0 if target >= 1. else 0. for target in iris.target ]

    learner = OgdLearner(LR(),Param())

    itertimes = 0
    while True:
        print itertimes
        cost = 0
        for i in xrange(len(y)):
            inst = Inst(zip(xrange(len(X[i])),X[i]), y[i])
            p = learner.predict(inst)
            inst_cost = logloss(p, inst.y)
            print i, p, inst.y, inst_cost
            cost += learner.model.cost(inst)
            learner.update(inst)
        itertimes += 1
        print 'total cost:', cost
        print 'average cost', cost/len(y)

