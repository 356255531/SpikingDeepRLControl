import nengo
import numpy as np
import matplotlib.pyplot as plt
from nengo.processes import WhiteSignal


class LoadFrom(nengo.solvers.Solver):
    def __init__(self, filename, weights=False):
        super(LoadFrom, self).__init__(weights=weights)
        self.filename = filename

    def __call__(self, A, Y, rng=None, E=None):
        if self.weights:
            shape = (A.shape[1], E.shape[1])
        else:
            shape = (A.shape[1], Y.shape[1])
        try:
            value = np.load(self.filename)
            assert value.shape == shape
        except IOError:
            value = np.zeros(shape)
        return value, {}


# helper to create the LoadFrom solver and the needed probe and do the saving
class WeightSaver(object):
    def __init__(self, connection, filename, sample_every=1.0, weights=False):
        assert isinstance(connection.pre, nengo.Ensemble)
        if not filename.endswith('.npy'):
            filename = filename + '.npy'
        self.filename = filename
        connection.solver = LoadFrom(self.filename, weights=weights)
        self.probe = nengo.Probe(connection, 'weights', sample_every=sample_every)
        self.connection = connection

    def save(self, sim):
        np.save(self.filename, sim.data[self.probe][-1].T)


model = nengo.Network(seed=1)
with model:
    stim = nengo.Node(lambda t: np.sin(t * np.pi * 2))
    a = nengo.Ensemble(100, 1)
    b = nengo.Ensemble(100, 1)
    conn = nengo.Connection(a, b, learning_rule_type=nengo.PES())
    error = nengo.Node(size_in=1)

    nengo.Connection(stim, a)
    nengo.Connection(a, error, transform=1)
    nengo.Connection(b, error, transform=-1)
    nengo.Connection(error, conn.learning_rule)

    ws = WeightSaver(conn, 'my_weights')  # add this line

with nengo.Simulator(model) as sim:
    sim.run(3)
    ws.save(sim)  # and add this line when you're done