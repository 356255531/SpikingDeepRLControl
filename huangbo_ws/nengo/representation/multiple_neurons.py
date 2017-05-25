import nengo
import numpy as np
import matplotlib.pyplot as plt


model = nengo.Network(label='Many Neurons')
with model:
    # Our ensemble consists of 100 leaky integrate-and-fire neurons,
    # representing a one-dimensional signal
    A = nengo.Ensemble(100, dimensions=1)

    sin = nengo.Node(lambda t: np.sin(8 * t))  # Input is a sine
    nengo.Connection(sin, A, synapse=0.01)  # 10ms filter

with model:
    sin_probe = nengo.Probe(sin)
    A_probe = nengo.Probe(A, synapse=0.01)  # 10ms filter
    A_spikes = nengo.Probe(A.neurons)  # Collect the spikes

# Create our simulator
with nengo.Simulator(model) as sim:
    # Run it for 1 second
    sim.run(1)

from nengo.utils.matplotlib import rasterplot

# Plot the decoded output of the ensemble
plt.figure()
plt.plot(sim.trange(), sim.data[A_probe], label="A output")
plt.plot(sim.trange(), sim.data[sin_probe], 'r', label="Input")
plt.xlim(0, 1)
plt.legend()

# Plot the spiking output of the ensemble
plt.figure()
rasterplot(sim.trange(), sim.data[A_spikes])  #sim.trange() : Create a vector of times matching probed data.
plt.xlim(0, 1);
plt.show()

print A_spikes
print type(A_spikes)