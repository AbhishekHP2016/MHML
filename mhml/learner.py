# -*- coding: utf-8 -*-

from math import sqrt

import numpy as np

from model import (
    LogisticRegression,
    FactorMachine,
    FieldFactorMachine,
    ismodel,
    Inst,
    WInst
)


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

        '''Predict based on its model.'''

        return self.model.predict(inst)

    def cost(self, inst):

        '''Compute cost based on its model'''

        return self.model.cost(inst)

    def update(self, inst):

        '''Update the model based on inst.

        This is the only update function users should use.
        '''

        model = self.model
        if isinstance(model, LogisticRegression):
            self.w1update(model.gradient(inst))

        elif ismodel(model, FactorMachine) or ismodel(model, FieldFactorMachine):
            g1, g2 = model.gradient(inst)
            if g1:
                self.w1update(g1)
            self.w2update(g2)

    def w1update(self, g):
        return

    def w2update(self, g):
        return


class OnlineGradientDescentLearner(Learner):

    '''The online gradient descent updates its weights coordinately.

    The adaptive learning rate for wi is alpha / (beta + \sum g_i^2).
    Update weights by the formula:

        w_i <= w_i - learning_rate * (g + L2*w_i + L1*sign(w_i))

    Arguments:
        model: An instance of LR, FM, FFM, GLR
        param: must contain alpha, beta, L1, L2, D, K if isfactormachine(model)
    '''

    def __init__(self, model, param):
        super(OnlineGradientDescentLearner, self).__init__(model, param)
        D = param.D

        if param.linear:
            self.n1 = np.zeros(D)
        if ismodel(model, FactorMachine) or ismodel(model, FieldFactorMachine):
            K = param.K
            self.n2 = np.zeros((D, K))

    def updateElement(self, g, w, n):
        generator = (g.iteritems()) if isinstance(g, dict) else g

        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        for idx, val in generator:
            learningrate = alpha / (beta + np.sqrt(n[idx]))
            flag = np.greater_equal(
                abs(w[idx] - learningrate * (val + L2 * w[idx])), learningrate * L1)
            sign = np.sign(w[idx])
            w[idx] = flag * (w[idx] - learningrate * (val + L2 * w[idx] + sign * L1))
            n[idx] += val ** 2

    def w1update(self, g):
        self.updateElement(g, self.model.w1, self.n1)

    def w2update(self, g):
        self.updateElement(g, self.model.w2, self.n2)


class FtrlProximalLearner(Learner):

    '''This class implements the algorithm of FTRL-Proximal.

    The reference:
    https://static.googleusercontent.com/media/research.google.com/zh-TW//pubs/archive/41159.pdf.

    Arguments:
        model: An instance of LR, FM, FFM, GLR
        param: must contain alpha, beta, L1, L2, D, K if isfactormachine(model)
    '''

    def __init__(self, model, param):
        super(FtrlProximalLearner, self).__init__(model, param)
        D = param.D

        if param.linear:
            self.n1 = np.zeros(D)
            self.z1 = np.zeros(D)

        if ismodel(model, FactorMachine) or ismodel(model, FieldFactorMachine):
            K = param.K
            self.n2 = np.zeros((D, K))
            self.z2 = np.zeros((D, K))

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

    def w1update(self, g):
        self.updateElement(g, self.model.w, self.n1, self.z1)

    def w2update(self, g):
        self.updateElement(g, self.model.w, self.n1, self.z2)


class StochasticGradientDescentLearner(Learner):

    '''This class implements the Stochastic Gradient Descent learner based on
    Online Gradient Descent learner.

    This class needs to load the whole dataset. For each iteration, it shuffles
    the data and update according to the order.

    The terminal is based on two criteria. One is iteration numbers and the other
    is based on the performance of the validation set.
    '''

    def __init__(self, model, param, Xtrain, ytrain,
                 Xval=None, yval=None, iters=None, sample_weights=None):
        if not Xval and not iters:
            print 'You need to specify the validation set or iteration numbers'
            exit(0)

        self.learner = OnlineGradientDescentLearner(model, param)
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xval = Xval
        self.yval = yval
        self.iters = iters
        self.sample_weights = sample_weights

    def predict(self, inst):
        return self.learner.predict(inst)

    def update(self, inst):
        self.learner.update(inst)

    def cost(self, inst):
        return self.learner.cost(inst)

    def iterate(self):
        Xtrain = self.Xtrain
        ytrain = self.ytrain
        sample_weights = self.sample_weights

        permutation = np.random.permutation(len(Xtrain))
        for idx in permutation:
            sample_weight = sample_weights[idx] if sample_weights else 1.
            inst = WInst(Xtrain[idx], ytrain[idx], sample_weight)
            self.update(inst)

    def train(self):
        iters = self.iters
        Xval = self.Xval
        yval = self.yval
        if iters:
            for i in xrange(iters):
                print 'iters:', i
                self.iterate()
        else:
            performance = 1e+300
            numiters = 0
            while True:
                numiters += 1
                self.iterate()
                tmp_performance = 0
                for (x_inst, y_inst) in zip(Xval, yval):
                    inst = Inst(x_inst, y_inst)
                    tmp_performance += self.cost(inst)
                if tmp_performance < performance:
                    performance = tmp_performance
                    print 'performance:', performance
                    print 'iters:', numiters
                else:
                    return numiters

    def test(self, Xtest):
        ypred = []
        for x_inst in Xtest:
            ypred.append(self.predict(Inst(x_inst, 1.)))
        return ypred
