# coding: utf-8

from abc import ABCMeta
from collections import defaultdict
from math import sqrt

import numpy as np

from model import (
    LogisticRegression,
    FactorMachine,
    FieldFactorMachine,
    GeneralizedLogisticRegression
)


def isfactormachine(model):

    '''return true if model is FM or FFM or (GLR and model.model is FM or FFM) '''

    r = (isinstance(model, FactorMachine) or
         isinstance(model, FieldFactorMachine) or
         (isinstance(model, GeneralizedLogisticRegression) and
          (isinstance(model.model, FactorMachine) or isinstance(model.model, FieldFactorMachine))))
    return r


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

        self.n1 = defaultdict(float)
        if isfactormachine(model):
            self.K = model.K
            self.n2 = defaultdict(lambda: np.zeros(self.K,))

    def predict(self, inst):
        return self.model.predict(inst)

    def cost(self, inst):
        return self.model.cost(self, inst)

    def update(self, inst):
        model = self.model
        if isinstance(model, LogisticRegression):
            self.w1update(model.gradient(inst), model.w)

        elif isfactormachine(model):

            g1, g2 = model.gradient(inst)
            w1 = model.w1
            w2 = model.w2

            if g1:
                self.w1update(g1, w1)
            self.w2update(g2, w2)


class OnlineGradientDescentLearner(Learner):

    '''The online gradient descent updates its weights coordinately.

    The adaptive learning rate for wi is alpha / (beta + \sum g_i^2).
    Update weights by the formula:

        w_i <= w_i - learning_rate * (g + L2*w_i + L1*sign(w_i))
    '''

    def __init__(self, model, param):
        super(OnlineGradientDescentLearner, self).__init__(model, param)

    def updateElement(self, g, w, n):
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        for idx, val in g:
            learningrate = alpha / (beta + sqrt(n[idx]))
            flag = not abs(w[idx] - learningrate * (val + L2 * w[idx])) <= learningrate * L1
            sign = np.sign(w[idx])
            w[idx] = w[idx] - learningrate * (val + L2 * w[idx] + sign * L1) * flag
            n[idx] += val ** 2

    def w1update(self, g, w):
        self.updateElement(g, w, self.n1)

    def w2update(self, g, w):
        self.updateElement(g, w, self.n2)


class FtrlProximalLearner(Learner):

    '''This class implements the algorithm of FTRL-Proximal.

    The reference:
    https://static.googleusercontent.com/media/research.google.com/zh-TW//pubs/archive/41159.pdf.

    '''

    def __init__(self, model, param):
        super(FtrlProximalLearner, self).__init__(model, param)
        self.z1 = defaultdict(float)
        self.z2 = defaultdict(lambda: np.zeros(self.K))

    def updateElement(self, g, w, n, z):
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        for idx, val in g:
            sigma = -sqrt(n[idx]) / alpha
            n[idx] += val**2
            sigma += sqrt(n[idx]) / alpha
            z[idx] += val - sigma * w[idx]
            sign = np.sign(z[idx])
            w[idx] = -sign * max(abs(z[idx]) - L1, 0) / ((beta + sqrt(n[idx])) / alpha + L2)

    def w1update(self, g, w):
        self.updateElement(g, w, self.n1, self.n2)

    def w2update(self, g, w):
        self.updateElement(g, w, self.n1, self.n2)
