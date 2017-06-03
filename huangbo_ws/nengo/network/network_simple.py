import numpy as np
import matplotlib.pyplot as plt

import nengo
from nengo.dists import Choice
from nengo.utils.functions import piecewise
from nengo.utils.ipython import hide_input


def test_integrators(net):
    with net:
        piecewise_f = piecewise({
            0: 0,
            0.2: 0.5,
            1: 0,
            2: -1,
            3: 0,
            4: 1,
            5: 0
        })
        piecewise_inp = nengo.Node(piecewise_f)
        nengo.Connection(piecewise_inp, net.pre_integrator.input)
        input_probe = nengo.Probe(piecewise_inp)
        pre_probe = nengo.Probe(net.pre_integrator.ensemble, synapse=0.01)
        post_probe = nengo.Probe(net.post_integrator.ensemble, synapse=0.01)
    with nengo.Simulator(net) as sim:
        sim.run(6)
    plt.figure()
    plt.plot(sim.trange(), sim.data[input_probe], color='k')
    plt.plot(sim.trange(), sim.data[pre_probe], color='b')
    plt.plot(sim.trange(), sim.data[post_probe], color='g')

net = nengo.Network(label="Two integrators")
with net:
    with nengo.Network() as pre_integrator:
        pre_input = nengo.Node(size_in=1)
        pre_ensemble = nengo.Ensemble(100, dimensions=1)
        nengo.Connection(pre_ensemble, pre_ensemble, synapse=0.1)
        nengo.Connection(pre_input, pre_ensemble, synapse=None, transform=0.1)
    with nengo.Network() as post_integrator:
        post_input = nengo.Node(size_in=1)
        post_ensemble = nengo.Ensemble(100, dimensions=1)
        nengo.Connection(post_ensemble, post_ensemble, synapse=0.1)
        nengo.Connection(
            post_input, post_ensemble, synapse=None, transform=0.1)
    nengo.Connection(pre_ensemble, post_input)
