"""
This example illustrates how a chain of ensembles can be trained end-to-end
using NengoDL.  The function is not particularly challenging
(:math:`f(x) = 2x`), this is just to illustrate how to apply
:meth:`.Simulator.train`.
"""

import nengo
import nengo_dl
import numpy as np
import tensorflow as tf

with nengo.Network(seed=0) as net:
    # these parameter settings aren't necessary, but they set things up in
    # a more standard machine learning way, for familiarity
    net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
    net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
    net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)
    net.config[nengo.Connection].synapse = None

    # connect up our input node, and 3 ensembles in series
    a = nengo.Node([0.5])
    b = nengo.Ensemble(30, 1)
    c = nengo.Ensemble(30, 1)
    d = nengo.Ensemble(30, 1)
    nengo.Connection(a, b)
    nengo.Connection(b, c)
    nengo.Connection(c, d)

    # define our outputs with a probe on the last ensemble in the chain
    p = nengo.Probe(d)

n_steps = 1  # the number of simulation steps we want to run our model for
mini_size = 100  # minibatch size



with nengo_dl.Simulator(net, minibatch_size=mini_size, seed=3, device="/cpu:0", step_blocks=n_steps) as sim:
    # create input/target data. this could be whatever we want, but here
    # we'll train the network to output 2x its input
    input_data = np.random.uniform(-1, 1, size=(10000, n_steps, 1))
    target_data = input_data * 2

    # train the model, passing `input_data` to our input node `a` and
    # `target_data` to our output probe `p`. we can use whatever TensorFlow
    # optimizer we want here.
    sim.train({a: input_data}, {p: target_data},
              tf.train.MomentumOptimizer(5e-2, 0.9), n_epochs=30)
    sim.save_params("/home/huangbo/SpikingDeepRLControl/nengo_deep_learning/checkpoints")

    print sim.loss({a: input_data}, {p: target_data}, "mse")

with nengo_dl.Simulator(net, minibatch_size=1, seed=3, device="/cpu:0", step_blocks=n_steps) as sim_1:
    sim_1.load_params("/home/huangbo/SpikingDeepRLControl/nengo_deep_learning/checkpoints")
    input_data = np.random.uniform(-1, 1, size=(1, n_steps, 1))
    print input_data
    print input_data.shape
    print sim_1.data[p]
