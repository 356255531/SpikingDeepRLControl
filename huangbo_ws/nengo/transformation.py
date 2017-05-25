#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 17:45:28 2017

@author: huangbo
"""

from nengo.processes import WhiteSignal

model = nengo.Network(label="NEF summary")
with model:
    input = nengo.Node(WhiteSignal(1, high=5), size_out=1)
    input_probe = nengo.Probe(input, )
    A = nengo.Ensemble(30, dimensions=1, max_rates=Uniform(80, 100))
    Asquare = nengo.Node(size_in=1)
    nengo.Connection(input, A)
    nengo.Connection(A, Asquare, function=np.square)
    A_spikes = nengo.Probe(A.neurons)
    Asquare_probe = nengo.Probe(Asquare, synapse=0.01)
    

with nengo.Simulator(model) as sim:
    sim.run(1)

plt.figure(figsize=(10, 3.5))
plt.subplot(1, 2, 1)
plt.plot(
    sim.trange(),
    sim.data[input_probe],
    label="Input signal")
plt.plot(
    sim.trange(),
    sim.data[Asquare_probe],
    label="Decoded esimate")
plt.plot(
    sim.trange(),
    np.square(sim.data[input_probe]),
    label="Input signal squared")
plt.legend(loc="best", fontsize='medium')
plt.xlabel("Time (s)")
plt.xlim(0, 1)

ax = plt.subplot(1, 2, 2)
rasterplot(sim.trange(), sim.data[A_spikes])
plt.xlim(0, 1)
plt.xlabel("Time (s)")
plt.ylabel("Neuron");
hide_input()