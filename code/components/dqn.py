import nengo
import numpy as np
import tensorflow as tf


class SNN:
    """
    Spiking neural network as the Q value function approximation:

    Member function:
        constructor(input_dim, output_dim, num_hidden_neuros, decoder)

        encoder_initialization()

        train_network(train_input, train_labels)

        predict(input)
    """

    def __init__(self, input_dim, output_dim, num_hidden_neuros, decoder):
        '''
        constructor:
            args:
                input_dim, numpy array
                output_dim, numpy array, same shape with action
                num_hidden_neuros, int
                decoder, string, path to save weights

            usage:
                    Init the SNN-based Q-Network
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_neuros = num_hidden_neuros
        self.decoder = decoder

    def encoder_initialization(self):
        '''
            usage:
                    Init the encoder
        '''

        rng = np.random.RandomState(1)
        encoders = rng.normal(size=(self.num_hidden_neuros, self.input_dim))
        return encoders

    def train_network(self, train_input, train_labels):
        '''
        args:
                train_input, numpy array (batch_size * input_dim)
                train_labels, numpy array (batch_size * output_dim)

        usage:
                do supervised learning with respect the given input and labels
        '''

        encoders = self.encoder_initialization()
        solver = nengo.solvers.LstsqL2(reg=0.01)

        model = nengo.Network(seed=3)
        with model:
            input_neuron = nengo.Ensemble(n_neurons=self.num_hidden_neuros,
                                          dimensions=self.input_dim,
                                          neuron_type=nengo.LIFRate(),
                                          intercepts=nengo.dists.Uniform(-1.0, 1.0),
                                          max_rates=nengo.dists.Choice([100]),
                                          encoders=encoders,
                                          )
            output = nengo.Node(size_in=self.output_dim)
            conn = nengo.Connection(input_neuron,
                                    output,
                                    synapse=None,
                                    eval_points=train_input,
                                    function=train_labels,
                                    solver=solver
                                    )
            #encoders_weights = nengo.Probe(input_neuron.encoders, "encoders_weights", sample_every=1.0)
            conn_weights = nengo.Probe(conn, 'weights', sample_every=1.0)

        with nengo.Simulator(model) as sim:
            sim.run(3)
        # save the connection weights after training
        np.save(self.decoder, sim.data[conn_weights][-1].T)

    def predict(self, input):
        '''
        args:
                input, numpy array (batch_size * input_dim)

        return:
                output, numpy array, Q-function of given input data (batch_size * output_dim)

        usage:
                Do prediction by feedforward with given input (in our case is state)
        '''
        encoders = self.encoder_initialization()

        try:
            decoder = np.load(self.decoder)
        except IOError:
            rng = np.random.RandomState(1)
            decoder = rng.normal(size=(self.num_hidden_neuros, self.output_dim))

        model = nengo.Network(seed=3)
        with model:
            input_neuron = nengo.Ensemble(n_neurons=self.num_hidden_neuros,
                                          dimensions=self.input_dim,
                                          neuron_type=nengo.LIFRate(),
                                          intercepts=nengo.dists.Uniform(-1.0, 1.0),
                                          max_rates=nengo.dists.Choice([100]),
                                          encoders=encoders,
                                          )
            output = nengo.Node(size_in=self.output_dim)
            conn = nengo.Connection(input_neuron.neurons,
                                    output,
                                    synapse=None,
                                    transform=decoder.T
                                    )
        with nengo.Simulator(model) as sim:
            _, acts = nengo.utils.ensemble.tuning_curves(input_neuron, sim, inputs=input)
        return np.dot(acts, sim.data[conn].weights.T)


class DQN_Flappy_Bird(object):
    """docstring for DQN_Flappy_Bird"""

    def __init__(self, joint_dim=1, dqn_start_learning_rate=10e-6):
        super(DQN_Flappy_Bird, self).__init__()
        # Extract input date
        self.action_num = 3 * joint_dim
        self.dqn_start_learning_rate = dqn_start_learning_rate

        # Create the network and set the input, output, and label
        self.input_layer, self.output_layer = self.build_network()

        # Cost
        self.actions_batch = tf.placeholder(tf.float32, [None, self.action_num])
        self.target_q_func_batch = tf.placeholder(tf.float32, [None])
        self.preditct_q_func = tf.reduce_sum(
            tf.matmul(self.output_layer, self.actions_batch), reduction_indices=1)
        self.cost = tf.reduce_mean(
            tf.square(self.target_q_func_batch - self.preditct_q_func)
        )

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(dqn_start_learning_rate).minimize(self.cost)

        # Create the session and saver
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        # Init all the variable
        self.sess.run(tf.initialize_all_variables())

    def build_network(self):
        def gen_weights_var(shape):
            inital = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(inital)

        def gen_bias_var(shape):
            inital = tf.constant(0.01, shape=shape)
            return tf.Variable(inital)

        def connect_conv2d(input, weights, stride):
            return tf.nn.conv2d(
                input,
                weights,
                [1, stride, stride, 1],
                padding='SAME',
                use_cudnn_on_gpu=True
            )

        def connect_activ_relu(input, bias):
            return tf.nn.relu(input + bias)

        def connect_max_pool_2x2(input):
            return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # 1st conv layer filter parameter
        weights_conv_1 = gen_weights_var([8, 8, 4, 32])
        bias_activ_conv_1 = gen_bias_var([32])

        # 2nd conv layer filter parameter
        weights_conv_2 = gen_weights_var([4, 4, 32, 64])
        bias_activ_conv_2 = gen_bias_var([64])

        # 3rd conv layer filter parameter
        weights_conv_3 = gen_weights_var([3, 3, 64, 64])
        bias_activ_conv_3 = gen_bias_var([64])

        # 4th fully connect net parameter
        weights_fc_layer_4 = gen_weights_var([1600, 512])
        bias_fc_layer_4 = gen_bias_var([512])

        # 5th fully connect net parameter
        weights_fc_layer_5 = gen_weights_var([512, 2])
        bias_fc_layer_5 = gen_bias_var([2])

        # input layer
        input_layer = tf.placeholder(tf.float32, [None, 80, 80, 4])

        # Convo layer 1
        output_conv_lay_1 = connect_conv2d(input_layer, weights_conv_1, 4)
        output_active_con_lay_1 = connect_activ_relu(output_conv_lay_1, bias_activ_conv_1)
        output_max_pool_layer_1 = connect_max_pool_2x2(output_active_con_lay_1)

        # Convo layer 2
        output_conv_lay_2 = connect_conv2d(output_max_pool_layer_1, weights_conv_2, 2)
        output_active_con_lay_2 = connect_activ_relu(output_conv_lay_2, bias_activ_conv_2)

        # Convo layer 3
        output_conv_lay_3 = connect_conv2d(output_active_con_lay_2, weights_conv_3, 1)
        output_active_con_lay_3 = connect_activ_relu(output_conv_lay_3, bias_activ_conv_3)
        output_reshape_layer_3 = tf.reshape(output_active_con_lay_3, [-1, 1600])  # Convo layer 3 reshape to fully connected net

        # Fully connect layer 4
        output_fc_layer_4 = tf.matmul(output_reshape_layer_3, weights_fc_layer_4)
        output_active_fc_layer_4 = connect_activ_relu(output_fc_layer_4, bias_fc_layer_4)

        # Output layer
        output_layer = tf.matmul(output_active_fc_layer_4, weights_fc_layer_5) + bias_fc_layer_5

        return input_layer, output_layer

    def predict(self, state):
        return self.sess.run(self.output_layer, feed_dict={self.input_layer: state})

    def save_weights(self, num_episode, saved_directory="saved_weights/"):
        self.saver.save(self.sess, saved_directory + "dqn_weights", global_step=num_episode)

    def load_weights(self, saved_directory="saved_weights"):
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
