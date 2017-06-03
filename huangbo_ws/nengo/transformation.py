import nengo
import numpy as np
import matplotlib.pyplot as plt

# Create a 'model' object to which we can add ensembles, connections, etc.
model = nengo.Network(label="Communications Channel")
with model:
    # Create an abstract input signal that oscillates as sin(t)
    # sin = nengo.Node(np.sin)
    input = nengo.Node(np.array([1, 2, 3]))
    # sin = nengo.Node([1])

    # Create the neuronal ensembles
    A = nengo.Ensemble(100, dimensions=3)
    B = nengo.Ensemble(100, dimensions=3)

    # Connect the input to the first neuronal ensemble
    nengo.Connection(input, A)

    # Connect the first neuronal ensemble to the second
    # (this is the communication channel)
    nengo.Connection(A, B, transform=-1)

with model:
    sin_probe = nengo.Probe(input)
    A_probe = nengo.Probe(A, synapse=.01)  # ensemble output
    B_probe = nengo.Probe(B, synapse=.01)

with nengo.Simulator(model) as sim:
    sim.run(2)

plt.figure(figsize=(9, 3))
plt.subplot(1, 3, 1)
plt.title("Input")
plt.plot(sim.trange(), sim.data[sin_probe])
plt.ylim(0, 1.2)
plt.subplot(1, 3, 2)
plt.title("A")
plt.plot(sim.trange(), sim.data[A_probe])
plt.ylim(0, 1.2)
plt.subplot(1, 3, 3)
plt.title("B")
plt.plot(sim.trange(), sim.data[B_probe])
plt.ylim(0, 1.2)
plt.show()