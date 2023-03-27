import numpy as np
import tensorflow as tf

np.random.seed(0)
_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))def get_logp(C, z, log)

def t2np(tensor):
    return tensor.numpy()

def get_logp(C, z, logdet_J):
    logp = C * _GCONST_ - 0.5*tf.reduce_sum(z**2, 1) + logdet_J
    return logp

def negative_log_likelihood(log_prob):
    return -tf.math.log_sigmoid(log_prob)

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())
