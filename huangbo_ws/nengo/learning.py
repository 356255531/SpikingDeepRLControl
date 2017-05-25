import matplotlib.pyplot as plt
import nengo
import numpy as np


from nengo_extras.data import load_mnist
from nengo_extras.vision import Gabor, Mask


def one_hot(labels, c=None):
    assert labels.ndim == 1
    n = labels.shape[0]
    c = len(np.unique(labels)) if c is None else c
    y = np.zeros((n, c))
    y[np.arange(n), labels] = 1
    return y


rng = np.random.RandomState(9)