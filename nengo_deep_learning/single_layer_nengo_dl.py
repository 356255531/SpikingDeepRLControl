import nengo
import nengo_dl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

input_shape = 28*28
output_shape = 10

with nengo.Network(seed=3) as model:
    #nengo_dl.configure_trainable(model)

    # these parameter settings aren't necessary, but they set things up in
    # a more standard machine learning way, for familiarity
    model.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
    model.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
    model.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)
    model.config[nengo.Connection].synapse = None

    # connect up our input node, and 3 ensembles in series
    input = nengo.Node(size_in=input_shape)
    layer_1 = nengo.Ensemble(n_neurons=1000, dimensions=input_shape)
    #layer_2 = nengo.Ensemble(n_neurons=200, dimensions=1)
    output = nengo.Node(size_in=output_shape)

    output_p = nengo.Probe(output)

    conn_1 = nengo.Connection(input, layer_1)
    conn_2 = nengo.Connection(layer_1, output)
    #conn_3 = nengo.Connection(layer_2, output)

    #model.config[conn_2].trainable = True

    conn_1_p = nengo.Probe(conn_1)
    conn_2_p = nengo.Probe(conn_2)
    #conn_3_p = nengo.Probe(conn_3)

n_steps = 1  # the number of simulation steps we want to run our model for
mini_size = 100  # minibatch size

with nengo_dl.Simulator(model, minibatch_size=mini_size, device="/gpu:0", step_blocks=n_steps) as sim:
    # create input/target data. this could be whatever we want, but here
    # we'll train the network to output 2x its input

    from keras.datasets import mnist
    from keras.utils import np_utils

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # data pre-processing
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
    y_train = np_utils.to_categorical(y_train, nb_classes=10)
    y_test = np_utils.to_categorical(y_test, nb_classes=10)

    input_data = np.expand_dims(X_train, axis=1)
    target_data = np.expand_dims(y_train, axis=1)

    sim.train({input: input_data}, # (batch_size, sim.step_blocks, node.size_out)
              {output_p: target_data}, # (batch_size, sim.step_blocks, probe.size_in)
              tf.train.MomentumOptimizer(5e-2, 0.9),
              n_epochs=1,
              objective="mse"
              )

    print sim.loss({input:input_data}, {output_p:target_data}, "mse")




