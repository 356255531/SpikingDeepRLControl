import nengo
import nengo_dl
import numpy as np
import tensorflow as tf
from copy import deepcopy



class DQN:
    '''
    Q_function approximation using Spiking neural network
    '''
    def __init__(self, input_shape, output_shape, batch_size_train, batch_size_predict):
        '''
        :param input_shape: the input shape of network, a number of integer 
        :param output_shape: the output shape of network, a number of integer
        :param batch_size_train: the batch size of training
        :param batch_size_predict: the batch size of prediction
        '''
        self.input_shape = input_shape
        self.output_shape = output_shape


        self.softlif_neurons = nengo_dl.SoftLIFRate(tau_rc=0.02, tau_ref=0.002, sigma=0.002)
        self.ens_params = dict(max_rates=nengo.dists.Choice([100]), intercepts=nengo.dists.Choice([0]))
        self.amplitude = 0.01

        self.model_train, self.sim_train, self.input_train, self.out_p_train = \
            self.build_simulator(minibatch_size=batch_size_train)

        self.model_predict, self.sim_predict, self.input_predict, self.out_p_predict = \
            self.build_simulator(minibatch_size=batch_size_predict)


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

    def build_simulator(self, minibatch_size):

        with nengo.Network(seed=0) as model:
            input, output = self.build_network()
            out_p = nengo.Probe(output)

        sim = nengo_dl.Simulator(model, minibatch_size=minibatch_size)
        return model, sim, input, out_p

    def choose_optimizer(self, opt, learning_rate=1):
        if opt == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif opt =='adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        elif opt == "rms":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif opt == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        return optimizer


    def objective(self, x, y):
        return tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)

    def training(self, train_whole_dataset, train_whole_labels, num_epochs):
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

        with self.model_train:
            nengo_dl.configure_trainable(self.model_train, default=True)

            train_inputs = {self.input_train: train_whole_dataset}
            train_targets = {self.out_p_train: train_whole_labels}

        optimizer = self.choose_optimizer('adadelta', 1)
        # construct the simulator
        self.sim_train.train(train_inputs,
                             train_targets,
                             optimizer,
                             n_epochs=num_epochs,
                             objective=self.objective
                             )

    def save(self, save_path):
        # save the parameters to file
        self.sim_train.save_params(save_path)

    def load_weights(self, flag, save_path):
        if flag == 'train':
            self.sim_train.load_params(save_path)
        elif flag == 'prediction':
            self.sim_predict.load_params(save_path)

    def predict(self, prediction_input):
        '''
        prediction of the network
        :param prediction_input: a input data shape = (minibatch_size, 1, input_shape)
        :return: prediction with shape = (minibatch_size, output_shape)
        '''

        input_data = {self.input_predict: prediction_input}
        self.sim_predict.step(input_feeds = input_data)
        #print self.sim_predict.data[self.out_p_predict].shape
        if self.sim_predict.data[self.out_p_predict].shape[1] == 1:
            output = self.sim_predict.data[self.out_p_predict]
            output = np.squeeze(output, axis=1)
        else:
            output = self.sim_predict.data[self.out_p_predict][:, -1, :]
        return deepcopy(output)

    def close_simulator(self):
        self.sim_train.close()
        self.sim_predict.close()




if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from tensorflow.examples.tutorials.mnist import input_data
    from sklearn.metrics import accuracy_score

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    X_test = mnist.test.images
    y_test = mnist.test.labels

    deep_qNetwork = DQN(input_shape=784,
                          output_shape=10,
                          batch_size_train=32,
                          batch_size_predict=10000
                          )

    deep_qNetwork.load_weights(flag='prediction', save_path='/home/huangbo/Desktop/weights/mnist_parameters')
    for i in range(10):
        #deep_qNetwork.load_weights(flag='train')
        # deep_qNetwork.training(train_whole_dataset = mnist.train.images[:, None, :],
        #                        train_whole_labels = mnist.train.labels[:, None, :],
        #                        num_epochs = 1
        #                        )
        # deep_qNetwork.save(save_path='/home/huangbo/Desktop/weights/mnist_parameters')

        test_input = X_test[:, None, :]
        prediction = deep_qNetwork.predict(prediction_input=test_input)
        acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(prediction, axis=1))
        print "the test acc is:", acc

    deep_qNetwork.close_simulator()