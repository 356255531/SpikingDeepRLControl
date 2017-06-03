import nengo
import numpy as np
import matplotlib.pyplot as plt



model = nengo.Network()
with model:
    pre = nengo.Ensemble(n_neurons=60, dimensions=1)
    input = nengo.Node([1])
    
    nengo.Connection(input, pre)

    
