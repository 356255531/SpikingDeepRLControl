import os
import h5py
import nengo
import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.metrics import accuracy_score


data_file = 'weights.h5'
input_shape = 784
output_shape = 10


data = np.zeros([1, 784])
label = np.zeros([1, 10])
nb_neuron = 1000

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


presentation_time = 0.1
model = nengo.Network(seed=3)
with model:

    input = nengo.Node(nengo.processes.PresentInput(X_train, presentation_time))
    pre = nengo.Ensemble(n_neurons=nb_neuron, dimensions=input_shape)
    post = nengo.Node(size_in=output_shape)
    error = nengo.Node(size_in=output_shape)
    output = nengo.Node(nengo.processes.PresentInput(y_train, presentation_time))

    nengo.Connection(input, pre)
    nengo.Connection(output, error, transform= -1)
    nengo.Connection(post, error, transform = 1)
    output_p = nengo.Probe(output, "output")


    if os.path.isfile(data_file):
        # data file with weights exists, so initialize learning connection with those weights
        with h5py.File(data_file, 'r') as hf:
            weights = np.array(hf.get('weights'))
        learn_conn = nengo.Connection(pre.neurons, post, transform=weights, learning_rule_type=nengo.PES())
        nengo.Connection(error, learn_conn.learning_rule)

    else:
        def init_func(x):
          return np.zeros(output_shape)
        learn_conn = nengo.Connection(pre, post, function=init_func, learning_rule_type=nengo.PES())
        nengo.Connection(error, learn_conn.learning_rule)

    conn_p = nengo.Probe(learn_conn, 'weights')

with nengo.Simulator(model) as sim:

  sim.run(time_in_seconds=60)
  weights = sim.data[conn_p][len(sim.trange()) - 1, :, :]
  _, acts = nengo.utils.ensemble.tuning_curves(pre, sim, inputs=X_test[0:100, :])

  print type(weights)
  print type(acts)

  print weights.shape
  print acts.shape

  output = np.dot(acts, weights.T)


  acc = accuracy_score(y_true=np.argmax(y_test[0:100,:], axis=1), y_pred=np.argmax(output[0:100,:],axis=1))
  print "the test acc is:", acc



