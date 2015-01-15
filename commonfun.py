# coding: utf-8

from math import exp, log

def logloss(p,y):
    return -y*log(max(p,1e-20)) - (1-y)*log(max(1-p,1e-20))


def sigmoid(z):
    return 1. / (1.+exp(-max(-35, min(35, z))))
