import nengo
import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils


class Q_network:

    def __init__(self, input_shape, output_shape, nb_hidden, weights_file):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.nb_hidden = nb_hidden
        self.weights_file = weights_file

    def build_netwokr(self, X_train, train_targets):

        encoders = np.random.normal(loc=0, scale=1, size=(self.nb_hidden, self.input_shape))

        ens_params = dict(eval_points=X_train,
                          neuron_type=nengo.LIFRate(),
                          intercepts=nengo.dists.Choice([-0.5]),
                          max_rates=nengo.dists.Choice([100]),
                          encoders=encoders,
                          )

        solver = nengo.solvers.LstsqL2(reg=0.01)

        with nengo.Network(seed=3) as model:
            neuron = nengo.Ensemble(n_neurons=self.nb_hidden, dimensions=self.input_shape, **ens_params)
            output = nengo.Node(size_in=self.output_shape)
            conn = nengo.Connection(neuron, output,
                                    synapse=None,
                                    eval_points=X_train,
                                    function=train_targets,
                                    solver=solver
                                    )

        with nengo.Simulator(model) as sim:
            _, acts = nengo.utils.ensemble.tuning_curves(neuron, sim, inputs=X_train)

        return np.dot(acts, sim.data[conn].weights.T)




