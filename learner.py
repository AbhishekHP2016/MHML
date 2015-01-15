# coding: utf-8 

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from math import sqrt


from model import LR, FM, FFM, GLR

class Learner(object):

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


class OgdLearner(Learner):
    '''
        The online gradient descent updates its weights coordinately.
        The adaptive learning rate for wi is alpha / (beta + \sum g_i^2)
    '''
    def __init__(self, model, param):
        super(OgdLearner, self).__init__(model,param)
        self.z = defaultdict(float)

    def update(self, inst):
        L1 = self.L1
        L2 = self.L2
        alpha = self.alpha
        beta = self.beta

        z = self.z
        model = self.model

        if isinstance(model, LR):
            g = model.gradient(inst)
            w = model.w
            for idx, val in g:
                learningrate = alpha / ( beta + sqrt(z[idx]) )
                if abs(w[idx] - learningrate * (val+L2*w[idx])) <= learningrate*L1:
                    w[idx] = 0
                else:
                    sign = 1. if w[idx] >= 0 else -1.
                    w[idx] -= learningrate * (val + L2*w[idx] + L1)
                z[idx] += val**2

        elif isinstance(model, FM) or (isinstance(model, GLR) and isinstance(model.model, FM)):
            # to do
            pass
        elif isinstance(model, FFM) or (isinstance(model, GLR) and isinstance(model.model, FFM)):
            # to do
            pass


#class FtrlProximalLearner(Learner):


