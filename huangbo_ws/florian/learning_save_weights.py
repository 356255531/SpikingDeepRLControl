import nengo
import numpy as np
import os
import h5py

data_file = 'weights.h5'
model = nengo.Network(seed=2)

with model:
  stim = nengo.Node(lambda t: np.sin(t))

  pre = nengo.Ensemble(n_neurons=100, dimensions=1)
  post = nengo.Ensemble(n_neurons=100, dimensions=1)

  nengo.Connection(stim, pre)

  error = nengo.Ensemble(n_neurons=100, dimensions=1)

  nengo.Connection(post, error, transform=1)
  nengo.Connection(pre, error, function=lambda x: x**2,  transform=-1)

  if os.path.isfile(data_file):
    # data file with weights exists, so initialize learning connection with those weights
    with h5py.File(data_file, 'r') as hf:
      weights = np.array(hf.get('weights'))
    learn_conn = nengo.Connection(pre.neurons, post, transform=weights, learning_rule_type=nengo.PES())
    nengo.Connection(error, learn_conn.learning_rule)
    # don't start learning right away
    stop_learn_input = 1
  else:
    # data file with weights does not exist, so initialize the learning connection with 0
    def init_func(x):
      return 0
    learn_conn = nengo.Connection(pre, post, function=init_func, learning_rule_type=nengo.PES())
    nengo.Connection(error, learn_conn.learning_rule)
    stop_learn_input = 0

  conn_p = nengo.Probe(learn_conn, 'weights')

  stop_learn = nengo.Node(stop_learn_input)
  nengo.Connection(stop_learn, error.neurons, transform=-10 * np.ones((100, 1)))

if __name__ == '__main__':
  sim = nengo.Simulator(model)

  sim.run(time_in_seconds=20)

  with h5py.File(data_file, 'w') as hf:
    hf.create_dataset('weights', data=sim.data[conn_p][len(sim.trange())-1,:,:], compression="gzip", compression_opts=9)


