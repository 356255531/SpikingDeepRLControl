import nengo
import numpy as np
from code.vision import Gabor, Mask
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.metrics import accuracy_score


class mnist_classification:

    def __init__(self, input_shape, output_shape, nb_hidden):
        '''
        Spiking neural network as the Q value function approximation
        :param input_shape: the input dimension without batch_size, example: state is 2 dimension, 
        action is 1 dimenstion, the input shape is 3.
        :param output_shape: the output dimension without batch_size, the dimenstion of Q values
        :param nb_hidden: the number of neurons in ensemble
        :param decoder: the path to save weights of connection channel
        '''
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.nb_hidden = nb_hidden


    def encoder_initialization(self):
        '''
        encoder is the connection relationship between input and the ensemble
        :return: initialised encoder
        '''
        rng = np.random.RandomState(9)
        encoders = Gabor().generate(self.nb_hidden, (1, 1), rng=rng)
        encoders = Mask((self.input_shape, 1)).populate(encoders, rng=rng, flatten=True)
        return encoders

    def traning_and_prediction(self, train_data, train_targets, evl_image):
        '''
        training the network useing all training data
        :param train_data: the training input, shape = (nb_samples, dim_samples)
        :param train_targets: the label or Q values shape=(nbm samples, dim_samples)
        :param simulation_time: the time to do the simulation, default = 100s
        :return: 
        '''

        #encoders = self.encoder_initialization()
        rng = np.random.RandomState(1)
        encoders = rng.normal(size=(1000, 28*28))
        solver = nengo.solvers.LstsqL2(reg=0.01)

        model = nengo.Network(seed=3)
        with model:
            input_neuron = nengo.Ensemble(n_neurons=self.nb_hidden,
                                          dimensions=self.input_shape,
                                          neuron_type=nengo.LIFRate(),
                                          intercepts=nengo.dists.Uniform(-1.0, 1.0),
                                          max_rates=nengo.dists.Choice([100]),
                                          encoders=encoders,
                                          )
            output = nengo.Node(size_in=self.output_shape)
            conn = nengo.Connection(input_neuron,
                                    output,
                                    synapse=None,
                                    eval_points=train_data,
                                    function=train_targets,
                                    solver=solver,
                                    #learning_rule_type=nengo.PES(learning_rate=1e-4, pre_tau=0.1)
                                    )
            self.ws = WeightSaver(conn, 'conn_weights')
        # training is done after create the simulator
        with nengo.Simulator(model) as sim:
            # prediction for a single image
            _, acts = nengo.utils.ensemble.tuning_curves(input_neuron, sim, inputs=evl_image)
            return np.dot(acts, sim.data[conn].weights.T)


class WeightSaver(object):
    def __init__(self, connection, filename, sample_every=None, weights=False):
        assert isinstance(connection.pre, nengo.Ensemble)
        if not filename.endswith('.npy'):
            filename = filename + '.npy'
        self.filename = filename
        #connection.solver = LoadFrom(self.filename, weights=weights)
        self.probe = nengo.Probe(connection, 'weights', sample_every=sample_every)
        self.connection = connection
    def save(self, sim):
        np.save(self.filename, sim.data[self.probe][-1].T)


if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # data pre-processing
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
    y_train = np_utils.to_categorical(y_train, nb_classes=10)
    y_test = np_utils.to_categorical(y_test, nb_classes=10)

    model = mnist_classification(input_shape=28*28, output_shape=10, nb_hidden=1000)

    # training
    prediction = model.traning_and_prediction(X_train, y_train, evl_image = X_test)
    #print prediction.shape
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(prediction, axis=1))
    print "the test acc is:", acc







