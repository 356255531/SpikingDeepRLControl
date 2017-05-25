import nengo
import numpy as np
import matplotlib.pyplot as plt

from nengo.dists import Uniform
from nengo.utils.ensemble import tuning_curves
from nengo.utils.ipython import hide_input
from nengo.utils.matplotlib import rasterplot

model = nengo.Network(label='A Single Neuron')
with model:
    neuron = nengo.Ensemble(100,dimensions=1)
with model:
    cos = nengo.Node([1])
    #cos = nengo.Node(lambda t: np.cos(8 * t))
    # Connect the input signal to the neuron
    nengo.Connection(cos, neuron)

with model:
    cos_probe = nengo.Probe(cos)  # The original input
    spikes = nengo.Probe(neuron.neurons)  # The raw spikes from the neuron
    voltage = nengo.Probe(neuron.neurons,'voltage')  # Subthreshold soma voltage of the neuron
    filtered = nengo.Probe(neuron, synapse=0.01)  # Spikes filtered by a 10ms post-synaptic filter

