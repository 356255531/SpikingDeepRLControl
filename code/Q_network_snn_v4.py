import os
import h5py
import nengo
import numpy as np


class Q_network:

    def __init__(self, input_shape, output_shape, nb_hidden, weights_path, presentation_time=0.1):
        '''
        Spiking neural network as the Q value function approximation
        :param input_shape: the input dimension without batch_size, example: state is 2 dimension, 
        action is 1 dimenstion, the input shape is 3.
        :param output_shape: the output dimension without batch_size, the dimenstion of Q values
        :param nb_hidden: the number of neurons in ensemble
        :param weights_path: the path to save weights of connection channel, e.g. ../weights.h5
        '''
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.nb_hidden = nb_hidden
        self.weights_path = weights_path
        self.presentation_time = presentation_time


    def train_network(self, train_data, train_targets):
        '''
        training the network useing all training data
        :param train_data: the training input, shape = (nb_samples, dim_samples)
        :param train_targets: the label or Q values shape=(nbm samples, dim_samples)
        :param simulation_time: the time to do the simulation, default = 100s
        :return: 
        '''

        self.model = nengo.Network(seed=1)

        with self.model:

            input = nengo.Node(nengo.processes.PresentInput(train_data, self.presentation_time))
            output = nengo.Node(nengo.processes.PresentInput(train_targets, self.presentation_time))

            self.pre = nengo.Ensemble(n_neurons=self.nb_hidden, dimensions=self.input_shape)
            post = nengo.Node(size_in=self.output_shape)
            error = nengo.Node(size_in=self.output_shape)


            nengo.Connection(input, self.pre)
            nengo.Connection(output, error, transform = -1)
            nengo.Connection(post, error, transform = 1)

            self.output = nengo.Probe(post)

            if os.path.isfile(self.weights_path):
                # data file with weights exists, so initialize learning connection with those weights
                weights = np.load(self.weights_path)
                print '------load the weights-------'
                learn_conn = nengo.Connection(self.pre.neurons, post, transform=weights, learning_rule_type=nengo.PES())
                nengo.Connection(error, learn_conn.learning_rule)

            else:
                def init_func(x):
                    return np.zeros(self.output_shape)

                print '------learning from random init-------'

                learn_conn = nengo.Connection(self.pre, post, function=init_func, learning_rule_type=nengo.PES())
                nengo.Connection(error, learn_conn.learning_rule)

            conn_p = nengo.Probe(learn_conn, 'weights')

        self.sim = nengo.Simulator(self.model)

        with  self.sim:
            # #self.sim.run_steps(train_data.shape[0])
            self.sim.run(int(train_data.shape[0]*0.1))

            weights = self.sim.data[conn_p][len(self.sim.trange()) - 1, :, :]
            np.save(self.weights_path, weights)


    def predict(self, evl_input):
        '''
        prediction after training, the output will be the corresponding q values
        :param evl_input: input must be a numpy array, system state and action paars, shape = (dim_sample)
        :return: the q values
        '''

        with self.sim:
            _, acts = nengo.utils.ensemble.tuning_curves(self.pre, self.sim, inputs=evl_input)
        return np.dot(acts, np.load(self.weights_path).T)


if __name__ == '__main__':
    from keras.datasets import mnist
    from keras.utils import np_utils
    from sklearn.metrics import accuracy_score

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # data pre-processing
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    print X_train.shape
    print y_train.shape


    model = Q_network(input_shape=28*28, output_shape=10, nb_hidden=1000,
                      weights_path='/home/huangbo/SpikingDeepRLControl/code/weights.npy')


    batch_size = 50
    idx = 0

    for i in range(10):
        # training
        model.train_network(X_train[idx:idx+batch_size, :], y_train[idx:idx+batch_size, :])

        #a = np.random.randint(0,10000)
        evl_point = X_test[0:100, :]
        #evl_point = evl_point[np.newaxis, :]

        prediction = model.predict(evl_point)


        acc = accuracy_score(np.argmax(y_test[0:100, :], axis=1), np.argmax(prediction, axis=1))
        print "the test acc is:", acc
        # print 'the gt:', np.argmax(gt)
        # print 'prediction',  np.argmax(prediction)

        idx = idx + batch_size
