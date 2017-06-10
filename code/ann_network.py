# from keras.layers import Dense
# from keras.models import Sequential
# from keras.backend import set_image_dim_ordering


# class Q_learning_network(object):

#     set_image_dim_ordering("tf")

#     def __init__(self, input_shape, output_shape, weights_file):
#         '''
#         The Q_value function approximation based on artificial neural network
#         :param input_shape: dimension of input
#         :param nb_classes: dimension of output
#         :param weights_file: path to save the weights
#         '''
#         self.input_shape = input_shape
#         self.output_shape = output_shape
#         self.weights_file = weights_file

#         self.network = self.build_model()

#     def build_model(self):
#         model = Sequential()
#         model.add(Dense(100, input_dim=self.input_shape, activation='relu'))
#         model.add(Dense(100, activation='relu'))
#         model.add(Dense(self.output_shape, activation='linear'))
#         return model

#     def train_network(self, training_data, label):
#         self.network.compile(optimizer="rmsprop",
#                              loss='mse',
#                              metrics=['accuracy'],
#                              )
#         self.network.train_on_batch(x=training_data, y=label)

#     def predict(self, input):
#         output = self.network.predict(input)
#         return output

#     def save_weights(self):
#         self.network.save_weights(self.weights_file)

#     def load_weights(self):
#         self.network.load_weights(self.weights_file)


# def main():
#     dqn = Q_learning_network(36, 3, "weights")


# if __name__ == '__main__':
#     main()


import tensorflow as tf
import numpy as np


class Q_learning_network(object):
    """docstring for Q_learning_network"""

    def __init__(self, action_num=3, dqn_start_learning_rate=10e-6):
        super(Q_learning_network, self).__init__()
        # Extract input date
        self.action_num = action_num
        self.dqn_start_learning_rate = dqn_start_learning_rate

        # Create the network and set the input, output, and label
        self.input_layer, self.output_layer = self.create_network()

        # Cost
        self.actions_batch = tf.placeholder(tf.float32, [None, self.action_num])
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

    def create_network(self):
        def gen_weights_var(shape):
            inital = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(inital)

        def gen_bias_var(shape):
            inital = tf.constant(0.01, shape=shape)
            return tf.Variable(inital)

        def connect_activ_relu(input, bias):
            return tf.nn.relu(input + bias)

        # 1st conv layer filter parameter
        weights_1 = gen_weights_var([36, 500])
        bias_activ_1 = gen_bias_var([500])

        # 2nd conv layer filter parameter
        weights_2 = gen_weights_var([500, 100])
        bias_activ_2 = gen_bias_var([100])

        # 3rd conv layer filter parameter
        weights_3 = gen_weights_var([100, 3])
        bias_activ_3 = gen_bias_var([3])

        # input layer
        input_layer = tf.placeholder(tf.float32, [None, 36])

        # hidden layer 1
        output_lay_1 = connect_activ_relu(tf.matmul(input_layer, weights_1), bias_activ_1)

        # hidden layer 2
        output_lay_2 = connect_activ_relu(tf.matmul(output_lay_1, weights_2), bias_activ_2)

        # Output layer
        output_layer = tf.matmul(output_lay_2, weights_3) + bias_activ_3

        return input_layer, output_layer

    def predict(self, state):
        return self.sess.run(self.output_layer, feed_dict={self.input_layer: state})

    def save_weights(self, num_episode, saved_directory="saved_weights/"):
        self.saver.save(self.sess, saved_directory + "dqn_weights", global_step=num_episode)

    def load_weights(self, saved_directory="saved_weights/"):
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
