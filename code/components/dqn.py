# import nengo
# import numpy as np


# class DQN:
#     """
#     Spiking neural network as the Q value function approximation:

#     Member function:
#         constructor(input_dim, output_dim, num_hidden_neuros, decoder)

#         encoder_initialization()

#         train_network(train_input, train_labels)

#         predict(input)
#     """

#     def __init__(self, input_dim, output_dim, num_hidden_neuros, decoder):
#         '''
#         constructor:
#             args:
#                 input_dim, numpy array
#                 output_dim, numpy array, same shape with action
#                 num_hidden_neuros, int
#                 decoder, string, path to save weights

#             usage:
#                     Init the SNN-based Q-Network
#         '''
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.num_hidden_neuros = num_hidden_neuros
#         self.decoder = decoder

#     def encoder_initialization(self):
#         '''
#             usage:
#                     Init the encoder
#         '''

#         rng = np.random.RandomState(1)
#         encoders = rng.normal(size=(self.num_hidden_neuros, self.input_dim))
#         return encoders

#     def train_network(self, train_input, train_labels):
#         '''
#         args:
#                 train_input, numpy array (batch_size * input_dim)
#                 train_labels, numpy array (batch_size * output_dim)

#         usage:
#                 do supervised learning with respect the given input and labels
#         '''

#         encoders = self.encoder_initialization()
#         solver = nengo.solvers.LstsqL2(reg=0.01)

#         model = nengo.Network(seed=3)
#         with model:
#             input_neuron = nengo.Ensemble(n_neurons=self.num_hidden_neuros,
#                                           dimensions=self.input_dim,
#                                           neuron_type=nengo.LIFRate(),
#                                           intercepts=nengo.dists.Uniform(-1.0, 1.0),
#                                           max_rates=nengo.dists.Choice([100]),
#                                           encoders=encoders,
#                                           )
#             output = nengo.Node(size_in=self.output_dim)
#             conn = nengo.Connection(input_neuron,
#                                     output,
#                                     synapse=None,
#                                     eval_points=train_input,
#                                     function=train_labels,
#                                     solver=solver
#                                     )
#             #encoders_weights = nengo.Probe(input_neuron.encoders, "encoders_weights", sample_every=1.0)
#             conn_weights = nengo.Probe(conn, 'weights', sample_every=1.0)

#         with nengo.Simulator(model) as sim:
#             sim.run(3)
#         # save the connection weights after training
#         np.save(self.decoder, sim.data[conn_weights][-1].T)

#     def predict(self, input):
#         '''
#         args:
#                 input, numpy array (batch_size * input_dim)

#         return:
#                 output, numpy array, Q-function of given input data (batch_size * output_dim)

#         usage:
#                 Do prediction by feedforward with given input (in our case is state)
#         '''
#         encoders = self.encoder_initialization()

#         try:
#             decoder = np.load(self.decoder)
#         except IOError:
#             rng = np.random.RandomState(1)
#             decoder = rng.normal(size=(self.num_hidden_neuros, self.output_dim))

#         model = nengo.Network(seed=3)
#         with model:
#             input_neuron = nengo.Ensemble(n_neurons=self.num_hidden_neuros,
#                                           dimensions=self.input_dim,
#                                           neuron_type=nengo.LIFRate(),
#                                           intercepts=nengo.dists.Uniform(-1.0, 1.0),
#                                           max_rates=nengo.dists.Choice([100]),
#                                           encoders=encoders,
#                                           )
#             output = nengo.Node(size_in=self.output_dim)
#             conn = nengo.Connection(input_neuron.neurons,
#                                     output,
#                                     synapse=None,
#                                     transform=decoder.T
#                                     )
#         with nengo.Simulator(model) as sim:
#             _, acts = nengo.utils.ensemble.tuning_curves(input_neuron, sim, inputs=input)
#         return np.dot(acts, sim.data[conn].weights.T)


# if __name__ == '__main__':
#     from keras.datasets import mnist
#     from keras.utils import np_utils
#     from sklearn.metrics import accuracy_score

#     (X_train, y_train), (X_test, y_test) = mnist.load_data()
#     # data pre-processing
#     X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
#     X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
#     y_train = np_utils.to_categorical(y_train, nb_classes=10)
#     y_test = np_utils.to_categorical(y_test, nb_classes=10)

#     model = Q_network(input_dim=28 * 28, output_dim=10, num_hidden_neuros=1000, decoder="/home/huangbo/Desktop/decoder.npy")

#     # training
#     model.train_network(X_train, y_train)
#     prediction = model.predict(X_test)

#     acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(prediction, axis=1))
#     print "the test acc is:", acc
import nengo
import nengo_dl
import numpy as np
import tensorflow as tf


class DQN:
    '''
    Q_function approximation using Spiking neural network
    '''

    def __init__(self, input_shape, output_shape, save_path):
        '''
        :param input_shape: the input shape of network, a number of integer 
        :param output_shape: the output shape of network, a number of integer 
        :param save_path: the path to save network parameters, in the prediction, network will load the weights in 
                this path.
                example: '/home/huangbo/Desktop/weights/mnist_parameters'
        '''
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.save_path = save_path

        self.softlif_neurons = nengo_dl.SoftLIFRate(tau_rc=0.02, tau_ref=0.002, sigma=0.002)
        self.ens_params = dict(max_rates=nengo.dists.Choice([100]), intercepts=nengo.dists.Choice([0]))
        self.amplitude = 0.01

    def build_network(self):
        # input_node
        input = nengo.Node(nengo.processes.PresentInput(np.zeros(shape=(1, self.input_shape)), 0.1))

        # layer_1
        x = nengo_dl.tensor_layer(input, tf.layers.dense, units=100)
        x = nengo_dl.tensor_layer(x, self.softlif_neurons, **self.ens_params)

        # layer_2
        x = nengo_dl.tensor_layer(x, tf.layers.dense, transform=self.amplitude, units=100)
        x = nengo_dl.tensor_layer(x, self.softlif_neurons, **self.ens_params)

        # output
        x = nengo_dl.tensor_layer(x, tf.layers.dense, units=self.output_shape)
        return input, x

    def choose_optimizer(self, opt, learning_rate=1):
        if opt == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif opt == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        elif opt == "rms":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif opt == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        return optimizer

    def objective(self, x, y):
        return tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)

    def training(self, minibatch_size, train_whole_dataset, train_whole_labels, num_epochs, pre_train_weights=None):
        '''
        Training the network, objective will be the loss function, default is 'mse', but you can alse define your
        own loss function, weights will be saved after the training. 
        :param minibatch_size: the batch size for training. 
        :param train_whole_dataset: whole training dataset, the nengo_dl will take minibatch from this dataset
        :param train_whole_labels: whole training labels
        :param num_epochs: how many epoch to train the whole dataset
        :param pre_train_weights: if we want to fine-tuning the network, load weights before training
        :return: None
        '''

        with nengo.Network(seed=0) as model:
            nengo_dl.configure_trainable(model, default=True)
            input, output = self.build_network()
            out_p = nengo.Probe(output)

            train_inputs = {input: train_whole_dataset}
            train_targets = {out_p: train_whole_labels}

        with nengo_dl.Simulator(model, minibatch_size=minibatch_size) as sim:

            if pre_train_weights != None:
                try:
                    sim.load_params(pre_train_weights)
                except:
                    pass

            optimizer = self.choose_optimizer('adadelta', 1)
            # construct the simulator
            sim.train(train_inputs, train_targets, optimizer, n_epochs=num_epochs, objective='mse')
            # save the parameters to file
            sim.save_params(self.save_path)

    def predict(self, prediction_input, minibatch_size=1):
        '''
        prediction of the network
        :param prediction_input: a input data shape = (minibatch_size, 1, input_shape)
        :param minibatch_size: minibatch size, default = 1
        :return: prediction with shape = (minibatch_size, output_shape)
        '''

        with nengo.Network(seed=0) as model:
            nengo_dl.configure_trainable(model, default=False)
            input, output = self.build_network()
            out_p = nengo.Probe(output)

        with nengo_dl.Simulator(model, minibatch_size=minibatch_size) as sim:
            try:
                sim.load_params(self.save_path)
            except:
                pass

            input_data = {input: prediction_input}
            sim.step(input_feeds=input_data)
            output = np.squeeze(sim.data[out_p], axis=1)

            return output


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from tensorflow.examples.tutorials.mnist import input_data
    from sklearn.metrics import accuracy_score

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    X_test = mnist.test.images
    y_test = mnist.test.labels

    deep_qNetwork = Deep_qNetwork_snn(input_shape=784,
                                      output_shape=10,
                                      save_path='saved_weights/'
                                      )

    for i in range(10):
        deep_qNetwork.training(minibatch_size=32,
                               train_whole_dataset=mnist.train.images[:, None, :],
                               train_whole_labels=mnist.train.labels[:, None, :],
                               num_epochs=1,
                               pre_train_weights='saved_weights/'
                               )

        test_input = X_test[:, None, :]
        prediction = deep_qNetwork.predict(prediction_input=test_input, minibatch_size=10000)
        acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(prediction, axis=1))
        print "the test acc is:", acc

    # test_input = X_test[156, :]
    # print test_input.shape
    #
    # test_input = test_input[np.newaxis, :]
    # test_input = test_input[np.newaxis, :]
    # print test_input.shape
    #
    # test_input_1 = np.squeeze(test_input, axis=0)
    # test_input_1 = np.squeeze(test_input_1, axis=0)
    # test_input_1 = np.reshape(test_input_1, newshape=(28, 28))
    #
    # prediction = deep_qNetwork.predict(prediction_input=test_input, minibatch_size=1)
    #
    # print prediction.shape
    # print np.argmax(prediction, axis=1)
    #
    # plt.figure()
    # plt.imshow(test_input_1)
    # plt.show()
