import nengo
import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils
from code.vision import Gabor, Mask
from sklearn.metrics import accuracy_score


class Q_network:

    def __init__(self, input_shape, output_shape, nb_hidden, decoder):
        '''
        :param input_shape: the input dimension without batch_size
        :param output_shape: the output dimension without batch_size
        :param nb_hidden: the number of neurons in ensemble
        :param decoder: the path to save weights of connection channel
        '''
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.nb_hidden = nb_hidden
        self.decoder = decoder

    def encoder_initialization(self, way="default"):
        if way=="random":
            encoders = np.random.normal(0, 1, size=(self.nb_hidden, self.input_shape))
        else:
            rng = np.random.RandomState(self.output_shape)
            encoders = Gabor().generate(self.nb_hidden, (self.output_shape, self.output_shape), rng=rng)
            encoders = Mask((28, 28)).populate(encoders, rng=rng, flatten=True)
        return encoders

    def train_network(self, train_data, train_targets, simulation_time=100):
        '''
        training the network useing all training data
        :param train_data: the training input, shape = (nb_samples, dim_samples)
        :param train_targets: the label or Q values shape=(nbm samples, dim_samples)
        :param simulation_time: the time to do the simulation, default = 100s
        :return: 
        '''

        encoders = self.encoder_initialization()
        solver = nengo.solvers.LstsqL2(reg=0.01)

        model = nengo.Network(seed=3)
        with model:
            input_neuron = nengo.Ensemble(n_neurons=self.nb_hidden,
                                           dimensions=self.input_shape,
                                           neuron_type=nengo.LIFRate(),
                                           intercepts=nengo.dists.Choice([-0.5]),
                                           max_rates=nengo.dists.Choice([100]),
                                           eval_points=train_data,
                                           encoders=encoders,
                                           )
            output = nengo.Node(size_in=self.output_shape)
            conn = nengo.Connection(input_neuron,
                                    output,
                                    synapse=None,
                                    eval_points=train_data,
                                    function=train_targets,
                                    solver=solver
                                    )
            conn_weights = nengo.Probe(conn, 'weights', sample_every=1.0)

        with nengo.Simulator(model) as sim:
            sim.run(simulation_time)

        # save the connection weights after training
        np.save(self.decoder, sim.data[conn_weights][-1].T)


    def predict(self, input):
        '''
        prediction after training, the output will be the corresponding q values
        :param input: system state and action shape = (dim_sample)
        :return: the q values
        '''
        encoders = self.encoder_initialization("random")
        #encoders = self.encoder_initialization()

        try:
            decoder = np.load(self.decoder)
        except IOError:
            decoder = np.zeros((self.nb_hidden, self.output_shape))

        model = nengo.Network(seed=3)
        with model:
            input_neuron = nengo.Ensemble(n_neurons=self.nb_hidden,
                                          dimensions=self.input_shape,
                                          neuron_type=nengo.LIFRate(),
                                          intercepts=nengo.dists.Choice([-0.5]),
                                          max_rates=nengo.dists.Choice([100]),
                                          encoders=encoders,
                                          )
            output = nengo.Node(size_in=self.output_shape)
            conn = nengo.Connection(input_neuron.neurons,
                                    output,
                                    synapse=None,
                                    transform=decoder.T
                                    )
        with nengo.Simulator(model) as sim:
            _, acts = nengo.utils.ensemble.tuning_curves(input_neuron, sim, inputs=input)

        return np.dot(acts, sim.data[conn].weights.T)

    def acc_calculation(self, test_data, test_label):
        encoders = self.encoder_initialization("random")
        print encoders.shape
        #encoders = self.encoder_initialization()

        model = nengo.Network(seed=3)
        with model:
            input_neuron = nengo.Ensemble(n_neurons=self.nb_hidden,
                                          dimensions=self.input_shape,
                                          neuron_type=nengo.LIFRate(),
                                          intercepts=nengo.dists.Choice([-0.5]),
                                          max_rates=nengo.dists.Choice([100]),
                                          encoders=encoders,
                                          )
            output = nengo.Node(size_in=self.output_shape)
            conn = nengo.Connection(input_neuron.neurons,
                                    output,
                                    synapse=None,
                                    transform=np.load(self.decoder).T
                                    )
        with nengo.Simulator(model) as sim:
            def get_outs(images):
                # The tuning curve tells us how each neuron responds to an incoming input signal.
                _, acts = nengo.utils.ensemble.tuning_curves(input_neuron, sim, inputs=images)
                return np.dot(acts, sim.data[conn].weights.T)

            def get_error(images, labels):
                return np.argmax(get_outs(images), axis=1) != labels

            prediction = get_outs(test_data)
            acc = accuracy_score(np.argmax(test_label, axis=1), np.argmax(prediction, axis=1))

            print "the test acc is:", acc



if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # data pre-processing
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
    y_train = np_utils.to_categorical(y_train, nb_classes=10)
    y_test = np_utils.to_categorical(y_test, nb_classes=10)

    model = Q_network(input_shape=28*28, output_shape=10, nb_hidden=1000, decoder="decoder.npy")

    # training
    model.train_network(X_train, y_train, simulation_time=100)

    # #single prediction
    # image = X_test[156,:]
    # image_new = np.reshape(image,(28,28))
    #
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(image_new)
    # plt.show()
    #
    # print "the predict number is:", np.argmax(model.predict(image))

    model.acc_calculation(X_test, y_test)



    print np.load("decoder.npy").shape










