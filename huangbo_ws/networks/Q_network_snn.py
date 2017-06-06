import nengo
import numpy as np
from code.vision import Gabor, Mask


class Q_network:

    def __init__(self, input_shape, output_shape, nb_hidden, decoder):
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
        self.decoder = decoder

    def encoder_initialization(self):
        '''
        encoder is the connection relationship between input and the ensemble
        :return: initialised encoder
        '''

        rng = np.random.RandomState(self.output_shape)
        encoders = Gabor().generate(self.nb_hidden, (1, 1), rng=rng)
        encoders = Mask((self.input_shape, 1)).populate(encoders, rng=rng, flatten=True)
        return encoders

    def train_network(self, train_data, train_targets):
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
            sim.run(1)
        # save the connection weights after training
        np.save(self.decoder, sim.data[conn_weights][-1].T)


    def predict(self, input):
        '''
        prediction after training, the output will be the corresponding q values
        :param input: input must be a numpy array, system state and action paars, shape = (dim_sample)
        :return: the q values
        '''
        encoders = self.encoder_initialization()

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
        sim = nengo.Simulator(model)
        _, acts = nengo.utils.ensemble.tuning_curves(input_neuron, sim, inputs=input)
        return np.dot(acts, sim.data[conn].weights.T)


if __name__ == '__main__':

    model = Q_network(input_shape=1, output_shape=1, nb_hidden=3000, decoder="decoder.npy")
    input_data = np.random.uniform(0, 1, size=(10000, 1))
    label = input_data * 2
    model.train_network(input_data, label)
    print "the predict number is:", model.predict(np.array(0.7))




    # from keras.datasets import mnist
    # from keras.utils import np_utils
    # from sklearn.metrics import accuracy_score
    #
    #
    #
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    #
    # # data pre-processing
    # X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
    # X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
    # y_train = np_utils.to_categorical(y_train, nb_classes=10)
    # y_test = np_utils.to_categorical(y_test, nb_classes=10)
    #
    # model = Q_network(input_shape=28*28, output_shape=10, nb_hidden=2000, decoder="decoder.npy")
    #
    # # training
    # model.train_network(X_train, y_train)
    #
    #
    # import timeit
    # start = timeit.default_timer()
    #
    # prediction = model.predict(X_test)
    # #print prediction.shape
    # acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(prediction, axis=1))
    # print "the test acc is:", acc
    #
    #
    # stop = timeit.default_timer()
    # print "the time is", stop - start
    #
    #
    # image = X_test[156,:]
    # image_new = np.reshape(image,(28,28))
    #
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(image_new)
    # plt.show()
    #
    # print "the predict number is:", np.argmax(model.predict(image))
