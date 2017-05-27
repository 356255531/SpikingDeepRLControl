import nengo
import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils
from vision import Gabor, Mask


class Q_network:

    def __init__(self, input_shape, output_shape, nb_hidden, encoder_file, weights_file):
        '''
        :param input_shape: the input dimension without batch_size
        :param output_shape: the output dimension without batch_size
        :param nb_hidden: the number of neurons in ensemble
        :param weights_file: the path to save weights of connection channel
        '''
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.nb_hidden = nb_hidden
        self.weights_file = weights_file
        self.encoder_file = encoder_file

    def encoder_initialization(self, way="default"):
        if way=="random":
            encoders = np.random.normal(0, 1, size=(self.nb_hidden, self.input_shape))
        else:
            rng = np.random.RandomState(self.output_shape)
            encoders = Gabor().generate(self.nb_hidden, (self.output_shape, self.output_shape), rng=rng)
            encoders = Mask((28, 28)).populate(encoders, rng=rng, flatten=True)
        return encoders

    def train_network(self, train_data, train_targets):

        encoders = self.encoder_initialization()
        solver = nengo.solvers.LstsqL2(reg=0.01)

        model = nengo.Network(seed=3)
        with model:
            input = nengo.Ensemble(n_neurons=self.nb_hidden,
                                   dimensions=self.input_shape,
                                   neuron_type=nengo.LIFRate(),
                                   intercepts=nengo.dists.Choice([-0.5]),
                                   max_rates=nengo.dists.Choice([100]),
                                   eval_points=train_data,
                                   encoders=encoders,
                                   )
            output = nengo.Node(size_in=self.output_shape)
            conn = nengo.Connection(input, output,
                                    synapse=None,
                                    eval_points=train_data,
                                    function=train_targets,
                                    solver=solver
                                    )

        with nengo.Simulator(model) as sim:
            _, acts = nengo.utils.ensemble.tuning_curves(input, sim, inputs=train_data)
            weights = sim.data[conn].weights.T
            #result = np.dot(acts, weights)
            np.save(self.weights_file, weights)
            np.save(self.encoder_file, encoders.T)

    def prediction(self, input):
        weights = np.load(self.weights_file)
        encoders = np.load(self.encoder_file)

        return np.dot(np.dot(input, encoders), weights)



if __name__ == '__main__':

    from keras.datasets import mnist
    from keras.utils import np_utils

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # data pre-processing
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
    y_train = np_utils.to_categorical(y_train, nb_classes=10)
    y_test = np_utils.to_categorical(y_test, nb_classes=10)

    model = Q_network(input_shape=28*28,
                      output_shape=10,
                      nb_hidden=1000,
                      encoder_file="encoders.npy",
                      weights_file="weights.npy")

    model.train_network(X_train, y_train)

    image = X_test[111,:]
    image_new = np.reshape(image,(28,28))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(image_new)
    plt.show()


    print np.argmax(model.prediction(image))









