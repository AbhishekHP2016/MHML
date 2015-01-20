# coding: utf-8

'''This file implements the some machine learning models'''

from abc import ABCMeta, abstractmethod
from mmh3 import hash
from collections import defaultdict, namedtuple
from itertools import combinations

import numpy as np

from util import sigmoid, logloss

Inst = namedtuple('Inst', ['x', 'y'])


class Param(object):

    def __init__(self, L1=0., L2=0.01, alpha=0.01, beta=1.):
        self.L1 = L1
        self.L2 = L2
        self.alpha = alpha
        self.beta = beta


class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, x):
        return

    @abstractmethod
    def gradient(self, data):
        return

    @abstractmethod
    def cost(self, data):
        return


class LogisticRegression(Model):

    '''This class implements the logistic regression. '''

    def __init__(self):
        self.w = defaultdict(int)

    def predict(self, inst):
        w = self.w

        wTx = 0
        for idx, val in inst.x:
            wTx += w[idx]*val

        return sigmoid(wTx)

    def cost(self, inst):
        p = self.predict(inst)
        return logloss(p, inst.y)

    def gradient(self, inst):
        x = inst.x
        y = inst.y

        p = self.predict(inst)
        return [(idx, (p-y)*val) for idx, val in x]


class FactorMachine(Model):

    '''This class implements the factor machine

    Argument:
        D: the dimension of feature space
        K: the dimension of the represented vector
    '''

    def __init__(self, param):

        K = param.K

        self.K = K
        self.w2 = defaultdict(lambda: np.random.normal(scale=0.05, size=(K,)))
        self.w1 = defaultdict(int) if param.linear else None

    def predict(self, inst):
        x = inst.x

        w2 = self.w2
        w1 = self.w1

        # total = (sum(i) wi) * (linearEnabled) + sum(i,j) fiTfj * vi * vj
        total = 0.
        total += sum(w1[idx]*val for idx, val in x) if w1 else 0.
        for (idx1, val1), (idx2, val2) in combinations(x, 2):
            total += np.inner(w2[idx1], w2[idx2]) * val1 * val2

        return total

    def cost(self, inst):
        return (inst.y - self.predict(inst))**2

    def gradient(self, inst):
        g1, g2 = self.dzdw(inst)
        diff = self.predict(inst.x) - inst.y

        g1 = [(idx, val*diff) for idx, val in g1] if g1 else None
        for v in g2.values():
            v *= diff

        return g1, g2

    def dzdw(self, inst):
        x = inst.x
        K = self.K
        w1 = self.w1
        w2 = self.w2

        g1 = [(idx, val) for idx, val in x] if w1 else None
        g2 = defaultdict(lambda: np.zeros((K,)))

        for (idx1, val1), (idx2, val2) in combinations(x, 2):
            key1, key2 = idxtokey()
            g2[key1] += val1 * val2 * w2[key2]
            g2[key2] += val1 * val2 * w2[key1]

        return g1, g2

    def idxtokey(idx1,idx2):
        return idx1,idx2

class FieldFactorMachine(Model):

    '''This class implements something called FFM.

    In FFM, we model the pairwise interaction for each category pair independently.

    Note: the hash trick is done inside the class.
    '''

    def __init__(self, param):
        K = param.K

        self.K = K
        self.D = param.D
        self.w1 = defaultdict(int) if param.linear else None
        self.w2 = defaultdict(lambda: np.random.normal(scale=0.05, size=(K,)))

    def predict(self, inst):

        w1 = self.w1
        w2 = self.w2

        x = inst.x

        total = 0
        if w1:
            for idx, val in x:
                total += w1[idx]*val

        for x1, x2 in combinations(x, 2):
            idx1, val1 = x1
            idx2, val2 = x2
            key1, key2 = self.idxtokey(idx1, idx2)
            total += np.inner(w2[key1], w2[key2]) * val1 * val2

        return total

    def cost(self, inst):
        return (inst.y - self.predict(inst))**2

    def gradient(self, inst):
        g1, g2 = self.dzdw(inst)
        diff = self.predict(inst.x) - self.y

        g1 = [(idx, val*diff) for idx, val in g1] if g1 else None

        for v in g2.values():
            v *= diff

        return g1, g2

    def dzdw(self, inst):
        x = inst.x

        K = self.K
        w1 = self.w1
        w2 = self.w2

        g2 = defaultdict(lambda: [0.] * K)
        for (), x2 in combinations(x, 2):
            idx1, val1 = x1
            idx2, val2 = x2
            key1, key2 = self.idxtokey(idx1, idx2)
            for k in xrange(K):
                g2[key1][k] += val1*val2*w2[key2][k]
                g2[key2][k] += val1*val2*w2[key1][k]

        g1 = [(idx, val) for idx, val in x] if w1 else None

        return g1, g2

    def idxtokey(self, idx1, idx2):
        D = self.D

        cat1key, cat1val = idx1
        cat2key, cat2val = idx2
        key1 = abs(hash(str(cat1key) + '_' + str(cat2key) + '_' + str(cat1val))) % D
        key2 = abs(hash(str(cat2key) + '_' + str(cat1key) + '_' + str(cat2val))) % D

        return key1, key2


class GeneralizedLogisticRegression(Model):

    '''This class implements generalized logistic regression.

    It regresses the logit with an arbitray function instead of the regular linear model.
    '''

    def __init__(self, model):
        self.model = model
        self.w1 = model.w1
        self.w2 = model.w2

    def predict(self, inst):
        z = self.model.predict(inst)
        return sigmoid(z)

    def cost(self, inst):
        return logloss(self.predict(inst), inst.y)

    def gradient(self, inst):

        model = self.model

        g1, g2 = model.dzdw()
        diff = self.predict(inst) - inst.y

        if g1:
            for i in xrange(len(g1)):
                idx, val = g1[i]
                g1[i] = idx, val*diff

        for v in g2.values():
            v *= diff

        return g1, g2

    @property
    def K(self):
        return self.model.K
