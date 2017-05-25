import nengo
import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils



(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize

y_train = np_utils.to_categorical(y_train, nb_classes=10)
y_test = np_utils.to_categorical(y_test, nb_classes=10)

print X_train.shape
print y_train.shape


# --- set up network parameters
input_shape = X_train.shape[1]  # input_shape=784
output_shape = y_train.shape[1]  # output_shape=10


n_hid = 1000
encoders = np.random.normal(0, 1, size= (1000, 28*28))

ens_params = dict(
    eval_points=X_train,
    neuron_type=nengo.LIFRate(),
    intercepts=nengo.dists.Choice([-0.5]),
    max_rates=nengo.dists.Choice([100]),
    encoders=encoders,
    )

solver = nengo.solvers.LstsqL2(reg=0.01)
# solver = nengo.solvers.LstsqL2(reg=0.0001)

with nengo.Network(seed=3) as model:
    a = nengo.Ensemble(n_neurons=n_hid, dimensions=input_shape, **ens_params)
    v = nengo.Node(size_in=output_shape)
    conn = nengo.Connection(
        a, v, synapse=None,
        eval_points=X_train, function=y_train, solver=solver)


with nengo.Simulator(model) as sim:
    def get_outs(images):
        _, acts = nengo.utils.ensemble.tuning_curves(a, sim, inputs=images)
        return np.dot(acts, sim.data[conn].weights.T)

    def get_error(images, labels):
        return np.argmax(get_outs(images), axis=1) != labels

    train_error = 100 * get_error(X_train, y_train).mean()
    test_error = 100 * get_error(X_test, y_test).mean()
    print("Train/test error: %0.2f%%, %0.2f%%" % (train_error, test_error))



