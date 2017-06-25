#from urllib.request import urlretrieve
import zipfile

import nengo
import nengo_dl
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from urllib2 import urlopen




# softlif parameters (lif parameters + sigma)
softlif_neurons = nengo_dl.SoftLIFRate(tau_rc=0.02, tau_ref=0.002, sigma=0.002)
# ensemble parameters
ens_params = dict(max_rates=nengo.dists.Choice([100]), intercepts=nengo.dists.Choice([0]))
# amplitude is used to scale the output of the nonlinearities (we set it to 1/max_rates
# so the output is scaled to ~1)
amplitude = 0.01
model =  nengo.Network(seed=0)


def build_network(neuron_type):
    # the input node that will be used to feed in input images
    inp = nengo.Node(nengo.processes.PresentInput(mnist.test.images, 0.1))

    # add the first convolutional layer
    x = nengo_dl.tensor_layer(
        inp,
        tf.layers.conv2d,
        shape_in=(28, 28, 1),
        filters=32,
        kernel_size=3
    )

    # apply the neural nonlinearity
    x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)

    # add another convolutional layer
    # note: we use the `amplitude` value to scale the output of the
    # previous neural layer
    x = nengo_dl.tensor_layer(
        x, tf.layers.conv2d,
        shape_in=(26, 26, 32),
        transform=amplitude,
        filters=32,
        kernel_size=3
    )
    x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)

    # add a pooling layer
    x = nengo_dl.tensor_layer(
        x,
        tf.layers.average_pooling2d,
        shape_in=(24, 24, 32),
        transform=amplitude,
        pool_size=2,
        strides=2
    )

    # add a dense (all-to-all connectivity) layer, with neural nonlinearity
    x = nengo_dl.tensor_layer(x, tf.layers.dense, units=128)
    x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)

    # add a dropout layer
    x = nengo_dl.tensor_layer(
        x,
        tf.layers.dropout,
        rate=0.4,
        transform=amplitude
    )

    # the final 10 dimensional class output
    x = nengo_dl.tensor_layer(x, tf.layers.dense, units=10)
    return inp, x


# construct the network
with model:
    # we'll make all the nengo objects in the network
    # non-trainable. we could train them if we wanted, but they don't
    # add any representational power so we can save some computation
    # by ignoring them. note that this doesn't affect the internal
    # components of tensornodes, which will always be trainable or
    # non-trainable depending on the code written in the tensornode.
    nengo_dl.configure_trainable(model, default=False)

    inp, out = build_network(softlif_neurons)
    out_p = nengo.Probe(out)

# construct the simulator
minibatch_size = 200
sim = nengo_dl.Simulator(model, minibatch_size=minibatch_size)

def objective(x, y):
    return tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)

opt = tf.train.AdadeltaOptimizer(learning_rate=1)


mnist = input_data.read_data_sets("MNIST_data/")
train_inputs = {inp: mnist.train.images[:, None, :]}
train_targets = {out_p: mnist.train.labels[:, None, :]}
test_inputs = {inp: mnist.test.images[:minibatch_size, None, :]}
test_targets = {out_p: mnist.test.labels[:minibatch_size, None, :]}

sim.train(train_inputs, train_targets, opt, objective=objective, n_epochs=5)
