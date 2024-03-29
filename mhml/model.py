# -*- coding: utf-8 -*-

'''This file implements the some machine learning models'''

from abc import ABCMeta, abstractmethod
from mmh3 import hash
from collections import defaultdict, namedtuple
from itertools import combinations

import numpy as np

from util import sigmoid, logloss

Inst = namedtuple('Inst', ['x', 'y'])
WInst = namedtuple('WInst', ['x', 'y', 'w'])

class Param(object):

    def __init__(self, L1=0., L2=0.01, alpha=0.01, beta=1., D=2**20, K=4, linear=True):
        self.L1 = L1
        self.L2 = L2
        self.alpha = alpha
        self.beta = beta
        self.D = D
        self.K = K
        self.linear = linear


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

    def __init__(self, D):
        self.w1 = np.zeros(D)

    def predict(self, inst):
        x = inst.x
        w = self.w1

        wTx = sum((w[idx] * val for idx, val in x))
        return sigmoid(wTx)

    def cost(self, inst):
        p = self.predict(inst)
        return logloss(p, inst.y)

    def gradient(self, inst):
        x = inst.x
        y = inst.y

        p = self.predict(inst)
        return [(idx, (p-y)*val) for idx, val in x]


class FactorMachineAbstract(Model):

    '''This class defines the abstract class of factor machines'''
    __metaclass__ = ABCMeta

    def __init__(self, param):

        K = param.K

        self.K = K
        self.w1 = np.zeros(param.D) if param.linear else None
        self.w2 = np.random.normal(scale=0.05, size=(param.D, K))

    def predict(self, inst):
        x = inst.x
        w1 = self.w1
        w2 = self.w2
        idxtolinearkey = self.idxtolinearkey
        idxtopairwisekey = self.idxtopairwisekey

        # total = (sum(i) wi) * (linearEnabled) + sum(i,j) fiTfj * vi * vj
        total = sum((w1[idxtolinearkey(idx)]*val for idx, val in x)) if w1 is not None else 0.
        for (idx1, val1), (idx2, val2) in combinations(x, 2):
            key1, key2 = idxtopairwisekey(idx1, idx2)
            total += np.inner(w2[key1], w2[key2]) * val1 * val2

        return total

    def cost(self, inst):
        return (inst.y - self.predict(inst))**2

    def gradient(self, inst):
        g1, g2 = self.dzdw(inst)
        w = inst.w if 'w' in dir(inst) else 1.  # inst is an Instance
        diff = (self.predict(inst) - inst.y) * w

        g1 = [(idx, val*diff) for idx, val in g1] if g1 is not None else None
        for v in g2.values():
            v *= diff

        return g1, g2

    def dzdw(self, inst):
        x = inst.x
        K = self.K
        w1 = self.w1
        w2 = self.w2
        idxtolinearkey = self.idxtolinearkey
        idxtopairwisekey = self.idxtopairwisekey

        g1 = [(idxtolinearkey(idx), val) for idx, val in x] if w1 is not None else None
        g2 = defaultdict(lambda: np.zeros((K,)))

        for (idx1, val1), (idx2, val2) in combinations(x, 2):
            if val1 and val2:
                key1, key2 = idxtopairwisekey(idx1, idx2)
                g2[key1] += val1 * val2 * w2[key2]
                g2[key2] += val1 * val2 * w2[key1]

        return g1, g2

    @abstractmethod
    def idxtopairwisekey(idx1, idx2):
        pass

    @abstractmethod
    def idxtolinearkey(idx):
        pass


class FactorMachine(FactorMachineAbstract):

    '''This class implements regular factor machines'''

    def __init__(self, param):
        super(FactorMachine, self).__init__(param)

    def idxtolinearkey(self, idx):
        return idx

    def idxtopairwisekey(self, idx1, idx2):
        return idx1, idx2


class FieldFactorMachine(FactorMachineAbstract):

    '''This class implements something called FFM.

    In FFM, we model the pairwise interaction for each category pair independently.

    Note: the hash trick is done inside the class.
    '''

    def __init__(self, param):
        self.D = param.D
        super(FieldFactorMachine, self).__init__(param)

    def idxtolinearkey(self, idx):
        D = self.D
        return abs(hash(str(idx))) % D

    def idxtopairwisekey(self, idx1, idx2):
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
        g1, g2 = model.dzdw(inst)
        diff = self.predict(inst) - inst.y

        if g1:
            g1 = [(idx, diff*val) for idx, val in g1]

        for v in g2.values():
            v *= diff

        return g1, g2

    @property
    def K(self):
        return self.model.K


def ismodel(model, _class):
    return (isinstance(model, _class) or
            (isinstance(model, GeneralizedLogisticRegression) and
             isinstance(model.model, _class)))
