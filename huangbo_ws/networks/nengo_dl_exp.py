import nengo
import nengo_dl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils import np_utils


# data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
y_train = np_utils.to_categorical(y_train, nb_classes=10)
y_test = np_utils.to_categorical(y_test, nb_classes=10)

X_train = X_train[:, np.newaxis, :]
X_test = X_test[:, np.newaxis, :]
y_train = y_train[:, np.newaxis, :]
y_test = y_test[:, np.newaxis, :]


model = nengo.Network()
model.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
model.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
model.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)
model.config[nengo.Connection].synapse = None


with model:
    rng = np.random.RandomState(9)
    encoders = rng.normal(size=(1000, 28*28))
    decoders = rng.normal(size=(10, 1000))

    input_node = nengo.Node(size_in=28*28)
    
    # neural ensembles
    layer = nengo.Ensemble(n_neurons=1000,
                           dimensions=28*28,
                           encoders=encoders
                           )


    # connect the input signals to ensemble inputs
    nengo.Connection(input_node, layer)
    
    # output node
    output_node = nengo.Node(size_in=10)

     # create a connection to compute (x+1)*y^2
    conn = nengo.Connection(layer, output_node, transform=decoders)
    model.config[conn].trainable = True

    # collect data on the inputs/outputs
    output_p = nengo.Probe(output_node)

inputs = {input_node: X_train}
targets = {output_p: y_train}

sim = nengo_dl.Simulator(model, minibatch_size=32, step_blocks=1, device="/gpu:0")
opt = tf.train.MomentumOptimizer(learning_rate=0.002, momentum=0.9, use_nesterov=True)
sim.train(inputs, targets, opt, n_epochs=100)



