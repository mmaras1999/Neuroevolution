import numpy as np
from scipy.special import expit

def ReLU(x):
    return np.max(x, 0.0)

def sigmoid(x):
    return expit(x)

def tanh(x):
    return np.tanh(x)
