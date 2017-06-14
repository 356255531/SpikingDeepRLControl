import nengo
import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils
from vision import Gabor, Mask
from gui import image_display_function

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
y_train = np_utils.to_categorical(y_train, nb_classes=10)
y_test = np_utils.to_categorical(y_test, nb_classes=10)

# -----------------------------------------------------------------#

rng = np.random.RandomState(9)
encoders = Gabor().generate(1000, (11, 11), rng=rng)
encoders = Mask((28, 28)).populate(encoders, rng=rng, flatten=True)

solver = nengo.solvers.LstsqL2(reg=0.01)

with nengo.Network(seed=3) as model:
    input = nengo.Node(nengo.processes.PresentInput(X_test, 0.1))
    layer = nengo.Ensemble(n_neurons=1000,
                          dimensions=784,
                          neuron_type=nengo.LIFRate(),
                          intercepts=nengo.dists.Choice([-0.5]),
                          max_rates=nengo.dists.Choice([100]),
                          encoders=encoders,
                                  )
    output = nengo.Node(size_in=10)
    nengo.Connection(input, layer)
    conn = nengo.Connection(layer, 
                            output, 
                            synapse=None,
                            eval_points=X_train, 
                            function=y_train, 
                            solver=solver
                            )
                            
                            
                            


                            