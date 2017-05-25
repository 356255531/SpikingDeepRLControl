import nengo
import numpy as np
import matplotlib.pyplot as plt

'''

This example demonstrates how to create a neuronal ensemble that will combine two 1-D inputs into one 2-D representation.

'''

def product(a,b):
    return a*b

model = nengo.Network(label='Combining')
with model:
    # Our input ensembles consist of 100 leaky integrate-and-fire neurons,
    # representing a one-dimensional signal
    A = nengo.Ensemble(100, dimensions=1)
    B = nengo.Ensemble(100, dimensions=1)

    # The output ensemble consists of 200 leaky integrate-and-fire neurons,
    # representing a two-dimensional signal
    output = nengo.Ensemble(200, dimensions=2, label='2D Population')
    
with model:
    # Create input nodes generating the sine and cosine
    sin = nengo.Node(output=np.sin)
    cos = nengo.Node(output=np.cos)
    
with model:
    nengo.Connection(sin, A)
    nengo.Connection(cos, B)

    # The square brackets define which dimension the input will project to
    nengo.Connection(A, output[1])
    nengo.Connection(B, output[0])
    
with model:
    sin_probe = nengo.Probe(sin)
    cos_probe = nengo.Probe(cos)
    A_probe = nengo.Probe(A, synapse=0.01)  # 10ms filter
    B_probe = nengo.Probe(B, synapse=0.01)  # 10ms filter
    out_probe = nengo.Probe(output, synapse=0.01)  # 10ms filter


# Create our simulator
with nengo.Simulator(model) as sim:
    # Run it for 5 seconds
    sim.run(5)


# Plot the decoded output of the ensemble
plt.figure()
plt.plot(sim.trange(), sim.data[out_probe][:, 0], 'b', label="2D output 0")
plt.plot(sim.trange(), sim.data[out_probe][:, 1], 'g', label="2D output 1")
plt.plot(sim.trange(), sim.data[A_probe], 'r', label="A output")
plt.plot(sim.trange(), sim.data[sin_probe], 'k', label="Sine")
plt.legend()
plt.show()

    
    