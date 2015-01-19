# coding: utf-8

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from math import sqrt

from model import LogisticRegression, FactorMachine, FieldFactorMachine, GeneralizedLogisticRegression


class Learner(object):

    __metaclass__ = ABCMeta

    def __init__(self, model, param):
        # model
        self.model = model

        # parameter
        self.alpha = param.alpha
        self.beta = param.beta
        self.L1 = param.L1
        self.L2 = param.L2

    def predict(self, inst):
        return self.model.predict(inst)

    def cost(self, inst):
        return self.model.cost(self, inst)

    @abstractmethod
    def update(self, inst):
        pass


class OnlineGradientDescentLearner(Learner):
    '''
        The online gradient descent updates its weights coordinately.
        The adaptive learning rate for wi is alpha / (beta + \sum g_i^2).
    '''
    def __init__(self, model, param):
        super(OgdLearner, self).__init__(model, param)
        self.z1 = defaultdict(float)
        if (isinstance(model, FactorMachine) or
                isinstance(model, FieldFactorMachine) or isinstance(model, GeneralizedLogisticRegression)):

            K = model.K if not GeneralizedLogisticRegression else model.model.K
            self.z2 = defaultdict(lambda: [0.] * K)

    def w1update(self, g, w):
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        z1 = self.z1
        for idx, val in g:
            learningrate = alpha / (beta + sqrt(z1[idx]))
            if abs(w[idx] - learningrate * (val+L2*w[idx])) <= learningrate*L1:
                w[idx] = 0
            else:
                sign = 1. if w[idx] >= 0 else -1.
                w[idx] -= learningrate * (val + L2*w[idx] + sign*L1)
            z1[idx] += val**2

    def w2update(self, g, w):
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        z2 = self.z2
        model = self.model
        K = model.K if isinstance(model, FactorMachine) or isinstance(model, FieldFactorMachine) else model.model.K

        for idx, val in g:
            for k in xrange(K):
                learningrate = alpha / (beta + sqrt(z2[idx][k]))
                if abs(w[idx][k] - learningrate * (val + L2 * w[idx][k])) <= learningrate*L1:
                    w[idx][k] = 0
                else:
                    sign = 1. if w[idx] >= 0 else -1.
                    w[idx][k] -= learningrate * (val + L2*w[idx][k] + L1 * sign)
                z2[idx] += val ** 2

    def update(self, inst):

        model = self.model
        if isinstance(model, LogisticRegression):
            self.w1update(model.gradient(inst), model.w)
        elif (isinstance(model, FactorMachine) or
                (isinstance(model, GeneralizedLogisticRegression) and isinstance(model.model, FactorMachine)) or
                isinstance(model, FieldFactorMachine) or
                (isinstance(model, GeneralizedLogisticRegression) and isinstance(model.model, FieldFactorMachine))):

            g1, g2 = model.gradient(inst)
            w1 = model.w1
            w2 = model.w2

            if g1:
                self.w1update(g1, w1)

            self.w2update(g2, w2)


#class FtrlProximalLearner(Learner):
