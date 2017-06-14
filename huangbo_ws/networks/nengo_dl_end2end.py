import nengo
import nengo_dl
import numpy as np
import tensorflow as tf

with nengo.Network(seed=0) as model:

    # these parameter settings aren't necessary, but they set things up in
    # a more standard machine learning way, for familiarity
    model.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
    model.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
    model.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)
    model.config[nengo.Connection].synapse = None

    # connect up our input node, and 3 ensembles in series
    a = nengo.Node([0.5])
    b = nengo.Ensemble(30, 1)
    c = nengo.Ensemble(30, 1)
    d = nengo.Ensemble(30, 1)
    conn1 = nengo.Connection(a, b)
    conn2 = nengo.Connection(b, c)
    conn3 = nengo.Connection(c, d)

    # define our outputs with a probe on the last ensemble in the chain
    p = nengo.Probe(d)

n_steps = 1  # the number of simulation steps we want to run our model for
mini_size = 100  # minibatch size

sim = nengo_dl.Simulator(model,
                        minibatch_size=mini_size,
                        step_blocks=n_steps,
                        device="/cpu:0")

input_data = np.random.uniform(-1, 1, size=(10000, n_steps, 1))
target_data = input_data * 2

sim.train({a: input_data},
          {p: target_data},
          tf.train.MomentumOptimizer(5e-2, 0.9),
          n_epochs=50
          )
sim.save_params("/home/huangbo/SpikingDeepRLControl/huangbo_ws/networks/saved_weights/model")
sim.close()


sim = nengo_dl.Simulator(model,
                        minibatch_size=1,
                        step_blocks=1,
                        device="/cpu:0"
                         )
sim.load_params("/home/huangbo/SpikingDeepRLControl/huangbo_ws/networks/saved_weights/model")
input = np.ones(shape=(1, 1, 1))

sim.step(input_feeds={a:input})
print sim.data[p]
sim.close()

# input = np.ones(shape=(100, 1, 1))
# print input.shape
# sim.step(input_feeds={a:input})
# print sim.data[p].shape
# print sim.data[p]
# sim.close()
