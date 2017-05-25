import matplotlib.pyplot as plt
import nengo
import numpy as np

model = nengo.Network(label='2D Representation')
with model:
    # Our ensemble consists of 100 leaky integrate-and-fire neurons,
    # and represents a 2-dimensional signal
    neurons = nengo.Ensemble(100, dimensions=2)

    # Create input nodes representing the sine and cosine
    sin = nengo.Node(output=np.sin)
    cos = nengo.Node(output=np.cos)

    # The indices in neurons define which dimension the input will project to
    nengo.Connection(sin, neurons[0])
    nengo.Connection(cos, neurons[1])

    sin_probe = nengo.Probe(sin, 'output')
    cos_probe = nengo.Probe(cos, 'output')
    neurons_probe = nengo.Probe(neurons, 'decoded_output', synapse=0.01)


with nengo.Simulator(model) as sim:
    # Run it for 5 seconds
    sim.run(5)

# Plot the decoded output of the ensemble
plt.figure()
plt.plot(sim.trange(), sim.data[neurons_probe], label="Decoded output")
plt.plot(sim.trange(), sim.data[sin_probe], 'r', label="Sine")
plt.plot(sim.trange(), sim.data[cos_probe], 'k', label="Cosine")
plt.legend()
plt.xlabel('time [s]');
plt.show()