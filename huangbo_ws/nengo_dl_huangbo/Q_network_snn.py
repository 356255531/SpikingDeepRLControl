import nengo
import nengo_dl
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class Deep_qNetwork_snn:
    def __init__(self, input_shape, output_shape, save_path):
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

        # layer_@
        x = nengo_dl.tensor_layer(x, tf.layers.dense, transform=self.amplitude, units=100)
        x = nengo_dl.tensor_layer(x, self.softlif_neurons, **self.ens_params)

        # output
        x = nengo_dl.tensor_layer(x, tf.layers.dense, units=self.output_shape)
        return input, x

    def objective(self, x, y):
        return tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)

    def classification_error(self, outputs, targets):
        return 100 * tf.reduce_mean(
            tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
                                 tf.argmax(targets[:, -1], axis=-1)),
                    tf.float32))

    def training(self, minibatch_size, train_whole_dataset, train_whole_labels, num_epochs, pre_train_weights=None):


        with nengo.Network(seed=0) as model:
            nengo_dl.configure_trainable(model, default=True)
            input, output = self.build_network()
            out_p = nengo.Probe(output)

            train_inputs = {input: train_whole_dataset}
            train_targets = {out_p: train_whole_labels}
            test_inputs = {input: mnist.test.images[:minibatch_size, None, :]}
            test_targets = {out_p: mnist.test.labels[:minibatch_size, None, :]}


        sim = nengo_dl.Simulator(model, minibatch_size=minibatch_size)

        print("error before training: %.1f%%" % sim.loss(test_inputs, test_targets, self.classification_error))

        if pre_train_weights!=None:
            sim.load_params(pre_train_weights)

        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=0.5)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)


        # construct the simulator
        sim.train(train_inputs, train_targets, optimizer, objective=self.objective, n_epochs=num_epochs)

        # save the parameters to file
        sim.save_params(self.save_path)


        print("error after training: %.1f%%" % sim.loss(test_inputs, test_targets, self.classification_error))

        sim.close()

    def predict(self, prediction_input):

        with nengo.Network(seed=0) as model:
            nengo_dl.configure_trainable(model, default=False)
            input, output = self.build_network()
            out_p = nengo.Probe(output)

        with nengo_dl.Simulator(model, minibatch_size=1) as sim:
            sim.load_params(self.save_path)
            input_data = {input: prediction_input}
            sim.step(input_feeds = input_data)
            print sim.data[out_p]
            return sim.data[out_p]




if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    deep_qNetwork = Deep_qNetwork_snn(input_shape=784,
                                      output_shape=10,
                                      save_path='/home/huangbo/Desktop/weights/mnist_parameters'
                                      )
    deep_qNetwork.training(minibatch_size=32,
                           train_whole_dataset = mnist.train.images[:, None, :],
                           train_whole_labels = mnist.train.labels[:, None, :],
                           num_epochs=10
                           )
    test_input = mnist.test.images[:1, None, :]
    print test_input.shape

    test_input_1 = np.squeeze(test_input, axis=0)
    test_input_1 = np.squeeze(test_input_1, axis=0)
    test_input_1 = np.reshape(test_input_1, newshape=(28, 28))

    prediction = deep_qNetwork.predict(prediction_input=test_input)
    print prediction.shape
    prediction = np.squeeze(prediction, axis=0)
    print np.argmax(prediction, axis=1)


    plt.figure()
    plt.imshow(test_input_1)
    plt.show()

