import nengo
import numpy as np
import tensorflow as tf
from copy import deepcopy
import nengo_dl


class SNN:
    '''
    Q_function approximation using Spiking neural network
    '''

    def __init__(self, dim, batch_size_train, batch_size_predict, opt, learning_rate, weiths_path="saved_weights_ann/"):
        '''
        :param input_shape: the input shape of network, a number of integer 
        :param output_shape: the output shape of network, a number of integer
        :param batch_size_train: the batch size of training
        :param batch_size_predict: the batch size of prediction
        '''
        self.input_shape = dim
        self.output_shape = 3 ** dim
        self.opt = opt
        self.learning_rate = learning_rate
        self.optimizer = self.choose_optimizer(self.opt, self.learning_rate)

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
        elif opt == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        elif opt == "rms":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif opt == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        return optimizer

    def objective(self, x, y):
        return tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)

    def train_network(self, train_whole_dataset, train_whole_labels, num_epochs):
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

        # construct the simulator
        self.sim_train.train(train_inputs,
                             train_targets,
                             optimizer,
                             n_epochs=num_epochs,
                             objective='mse'
                             )

    def save(self, save_path):
        # save the parameters to file
        self.sim_train.save_params(save_path + "dqn_weights")

    def load_weights(self, flag, save_path):
        try:
            if flag == 'train':
                self.sim_train.load_params(save_path)
            elif flag == 'prediction':
                self.sim_predict.load_params(save_path)
        except:
            print "No Weights loaded to " + flag

    def predict(self, prediction_input):
        '''
        prediction of the network
        :param prediction_input: a input data shape = (minibatch_size, 1, input_shape)
        :return: prediction with shape = (minibatch_size, output_shape)
        '''
        self.load_weights
        input_data = {self.input_predict: prediction_input}
        self.sim_predict.step(input_feeds=input_data)
        # print self.sim_predict.data[self.out_p_predict].shape
        if self.sim_predict.data[self.out_p_predict].shape[1] == 1:
            output = self.sim_predict.data[self.out_p_predict]
            output = np.squeeze(output, axis=1)
        else:
            output = self.sim_predict.data[self.out_p_predict][:, -1, :]
        return deepcopy(output)

    def close_simulator(self):
        self.sim_train.close()
        self.sim_predict.close()


class ANN(object):
    """docstring for Q_learning_network"""

    def __init__(self, joint_dim=1, dqn_start_learning_rate=10e-6):
        super(ANN, self).__init__()
        # Extract input date
        self._joint_dim = joint_dim
        self._action_num = 3 ** joint_dim
        self._dqn_start_learning_rate = dqn_start_learning_rate

        # Create the network and set the input, output, and label
        self.input_layer, self.output_layer = self.build_network(self._joint_dim, self._action_num)

        # Cost
        self.actions_batch = tf.placeholder(tf.float32, [None, self._action_num])
        self.target_q_func_batch = tf.placeholder(tf.float32, [None])
        self.preditct_q_func = tf.reduce_sum(
            tf.matmul(self.output_layer, tf.transpose(self.actions_batch)), reduction_indices=1)
        self.cost = tf.reduce_mean(
            tf.square(self.target_q_func_batch - self.preditct_q_func)
        )

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(dqn_start_learning_rate).minimize(self.cost)

        # Create the session and saver
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        # Init all the variable
        self.sess.run(tf.global_variables_initializer())

    def build_network(self, input_dim, output_dim):
        def gen_weights_var(shape):
            inital = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(inital)

        def gen_bias_var(shape):
            inital = tf.constant(0.01, shape=shape)
            return tf.Variable(inital)

        def connect_activ_relu(input, bias):
            return tf.nn.relu(input + bias)

        # 1st conv layer filter parameter
        weights_1 = gen_weights_var([input_dim, 500])
        bias_activ_1 = gen_bias_var([500])

        # 2nd conv layer filter parameter
        weights_2 = gen_weights_var([500, 100])
        bias_activ_2 = gen_bias_var([100])

        # 3rd conv layer filter parameter
        weights_3 = gen_weights_var([100, output_dim])
        bias_activ_3 = gen_bias_var([output_dim])

        # input layer
        input_layer = tf.placeholder(tf.float32, [None, input_dim])

        # hidden layer 1
        output_lay_1 = connect_activ_relu(tf.matmul(input_layer, weights_1), bias_activ_1)

        # hidden layer 2
        output_lay_2 = connect_activ_relu(tf.matmul(output_lay_1, weights_2), bias_activ_2)

        # Output layer
        output_layer = tf.matmul(output_lay_2, weights_3) + bias_activ_3

        return input_layer, output_layer

    def predict(self, state):
        return self.sess.run(self.output_layer, feed_dict={self.input_layer: state})

    def save_weights(self, num_episode, saved_directory="saved_weights_ann/"):
        self.saver.save(self.sess, saved_directory + "dqn_weights", global_step=num_episode)

    def load_weights(self, saved_directory="saved_weights_ann/"):
        checkpoint = tf.train.get_checkpoint_state(saved_directory)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Weights successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find pre-trained network weights")

    def train_network(self, states_batch, actions_batch, target_q_func_batch):
        # pdb.set_trace()
        _, cost = self.sess.run([self.optimizer, self.cost], feed_dict={
            self.input_layer: states_batch,
            self.target_q_func_batch: target_q_func_batch,
            self.actions_batch: actions_batch})
        return cost


# if __name__ == '__main__':
#     from keras.datasets import mnist
#     from keras.utils import np_utils
#     from sklearn.metrics import accuracy_score
#
#     (X_train, y_train), (X_test, y_test) = mnist.load_data()
#     # data pre-processing
#     X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
#     X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
#     y_train = np_utils.to_categorical(y_train, nb_classes=10)
#     y_test = np_utils.to_categorical(y_test, nb_classes=10)
#
#
#     model = Q_network(input_dim=28*28, output_dim=10, num_hidden_neuros=1000, decoder="/home/huangbo/Desktop/decoder.npy")
#
#     # training
#     model.train_network(X_train, y_train)
#     prediction = model.predict(X_test)
#
#     acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(prediction, axis=1))
#     print "the test acc is:", acc


def main():
    dqn = DQN(1, 3, num_hidden_neuros=1000, decoder="decoder.npy")
    print dqn.predict(np.array([0]))

if __name__ == '__main__':
    main()
