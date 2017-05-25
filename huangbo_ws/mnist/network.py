import nengo
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.backend import set_image_dim_ordering
from batch_generator import batch_generator
import matplotlib.pyplot as plt
import numpy as np
from nengo.processes import WhiteSignal

model = nengo.Network(label="Delayed connection")

with model:
    # We'll use white noise as input
    training_generator = batch_generator(1, "training")
    image, label = training_generator.next()

    image = np.squeeze(image)
    label = np.squeeze(label)

    # print image.shape
    # print label.shape

    input = nengo.Node(image, size_out=28*28, label="input")
    neurons = nengo.Ensemble(100, dimensions=28*28)
    nengo.Connection(input, neurons, transform=2)

    input_porbe = nengo.Probe(input)
    neurons_probe = nengo.Probe(neurons)


with nengo.Simulator(model) as sim:
    # Run it for 5 seconds
    sim.run(5)


# Plot the decoded output of the ensemble
plt.figure()
plt.plot(sim.trange(), sim.data[neurons_probe], label="output")
plt.plot(sim.trange(), sim.data[input_porbe], 'r', label="input")
plt.legend()
plt.xlabel('time [s]')
plt.show()

