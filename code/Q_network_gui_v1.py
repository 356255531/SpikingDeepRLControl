import nengo
import numpy as np


from vision import Gabor, Mask
from gui import image_display_function
from keras.datasets import mnist
from keras.utils import np_utils


# -------------------------------- load data -----------------------------#
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
y_train = np_utils.to_categorical(y_train, nb_classes=10)
y_test = np_utils.to_categorical(y_test, nb_classes=10)

rng = np.random.RandomState(9)

# --- set up network parameters
n_vis = X_train.shape[1]
n_out = y_train.shape[1]
n_hid = 1000

# encoders = rng.normal(size=(n_hid, 11, 11))
encoders = Gabor().generate(n_hid, (11, 11), rng=rng)
encoders = Mask((28, 28)).populate(encoders, rng=rng, flatten=True)

ens_params = dict(eval_points=X_train,
                  neuron_type=nengo.LIFRate(),
                  intercepts=nengo.dists.Choice([-0.5]),
                  max_rates=nengo.dists.Choice([100]),
                  encoders=encoders,
                  )

solver = nengo.solvers.LstsqL2(reg=0.01)
presentation_time = 0.1

with nengo.Network(seed=3) as model:
    u = nengo.Node(nengo.processes.PresentInput(X_test, presentation_time))
    a = nengo.Ensemble(n_hid, n_vis, **ens_params)
    v = nengo.Node(size_in=n_out)
    nengo.Connection(u, a, synapse=None)
    conn = nengo.Connection(a, v, synapse=None,
                            eval_points=X_train, 
                            function=y_train, 
                            solver=solver
                            )

    # --- image display
    image_shape = (1, 28, 28)
    display_f = image_display_function(image_shape, offset=1, scale=128)
    display_node = nengo.Node(display_f, size_in=u.size_out)
    nengo.Connection(u, display_node, synapse=None)

    # --- output spa display
    vocab_names = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR',
                   'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE']
    vocab_vectors = np.eye(len(vocab_names))

    vocab = nengo.spa.Vocabulary(len(vocab_names))
    for name, vector in zip(vocab_names, vocab_vectors):
        vocab.add(name, vector)

    config = nengo.Config(nengo.Ensemble)
    config[nengo.Ensemble].neuron_type = nengo.Direct()
    with config:
        output = nengo.spa.State(len(vocab_names), subdimensions=10, vocab=vocab)
    nengo.Connection(v, output.input)
