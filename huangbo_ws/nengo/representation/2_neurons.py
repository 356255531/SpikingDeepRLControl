import nengo
import numpy as np
import matplotlib.pyplot as plt

from nengo.dists import Uniform

model = nengo.Network(label='Two Neurons')
with model:
    neurons = nengo.Ensemble(
        2, dimensions=1,  # Representing a scalar
        intercepts=Uniform(-.5, -.5),  # Set the intercepts at .5
        max_rates=Uniform(100, 100),  # Set the max firing rate at 100hz
        encoders=[[1], [-1]])  # One 'on' and one 'off' neuron


    sin = nengo.Node(lambda t: np.sin(8 * t))
    nengo.Connection(sin, neurons, synapse=0.01)

    sin_probe = nengo.Probe(sin)  # The original input
    spikes = nengo.Probe(neurons.neurons)  # Raw spikes from each neuron
    voltage = nengo.Probe(
        neurons.neurons, 'voltage')  # Subthreshold soma voltages of the neurons
    filtered = nengo.Probe(
        neurons, synapse=0.01)  # Spikes filtered by a 10ms post-synaptic filter


with nengo.Simulator(model) as sim:  # Create a simulator
    sim.run(1)  # Run it for 1 second


# Plot the decoded output of the ensemble
plt.figure()
plt.plot(sim.trange(), sim.data[filtered])
plt.plot(sim.trange(), sim.data[sin_probe])
plt.xlim(0, 1)

plt.show()

# Plot the spiking output of the ensemble
from nengo.utils.matplotlib import rasterplot
plt.figure(figsize=(10, 8))
plt.subplot(221)
rasterplot(sim.trange(), sim.data[spikes], colors=[(1, 0, 0), (0, 0, 0)])
plt.xlim(0, 1)
plt.yticks((0, 1), ("On neuron", "Off neuron"))

# Plot the soma voltages of the neurons
plt.subplot(222)
plt.plot(sim.trange(), sim.data[voltage][:, 0] + 1, 'r')
plt.plot(sim.trange(), sim.data[voltage][:, 1], 'k')
plt.yticks(())
plt.axis([0, 1, 0, 2])
plt.subplots_adjust(wspace=0.05)

plt.show()