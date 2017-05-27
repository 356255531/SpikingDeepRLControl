import nengo
import numpy as np
from nengo.processes import WhiteSignal

input_shape = 1
output_shape = 1
n_hid = 100
X_train = np.sin(2*np.pi)
train_targets = 1
encoders = np.random.normal(0, 1, size= (100, input_shape))
solver = nengo.solvers.LstsqL2(reg=0.01)

model = nengo.Network(seed=3)

with model:
    a = nengo.Ensemble(n_neurons=n_hid,
                       dimensions=input_shape,
                       eval_points=X_train,
                       neuron_type=nengo.LIFRate(),
                       intercepts=nengo.dists.Choice([-0.5]),
                       max_rates=nengo.dists.Choice([100]),
                       encoders=encoders,
                       )
    v = nengo.Node(size_in=output_shape)
    conn = nengo.Connection(
        a, v, synapse=None,
        eval_points=X_train,
        function=train_targets,
        solver=solver)