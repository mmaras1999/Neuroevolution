import numpy as np
from math import exp
from scipy.special import expit

def ReLU(x):
    return np.max(x, 0.0)

def sigmoid(x):
    return expit(x)

def sigmoid_3(x): 
    if x < 0:
        return 1 - 1/(1 + exp(3 * x))
    else:
        return 1/(1 + exp(-3 * x))

def tanh(x):
    return np.tanh(x)
