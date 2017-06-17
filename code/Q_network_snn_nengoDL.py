import nengo
import nengo_dl
import numpy as np
import tensorflow as tf

class Spiking_Qnetwork:

    def __init__(self, input_shape, output_shape, nb_hidden, weights_path):
        '''
        Spiking neural network as the Q value function approximation
        :param input_shape: the input dimension without batch_size, example: state is 2 dimension, 
        action is 1 dimenstion, the input shape is 3.
        :param output_shape: the output dimension without batch_size, the dimenstion of Q values
        :param nb_hidden: the number of neurons in ensemble
        :param weights_path: the path to save all parameters
        '''
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.nb_hidden = nb_hidden
        self.weights_path = weights_path

        self.model = self.build()

    def cross_entropy(self, prediction, label):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=prediction))

    def encoder_decoder_initialization(self, shape):
        '''
        :return: initialised encoder or decoder
        '''
        rng = np.random.RandomState(1)
        coders = rng.normal(size=shape)
        return coders

    def build(self):
        encoders = self.encoder_decoder_initialization(shape=(self.nb_hidden, self.input_shape))
        decoders = self.encoder_decoder_initialization(shape=((self.nb_hidden, self.output_shape)))

        model = nengo.Network(seed=3)
        with model:
            nengo_dl.configure_trainable(model, default=True)
            model.config[nengo.Ensemble].neuron_type = nengo_dl.neurons.SoftLIFRate()
            model.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
            model.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)
            model.config[nengo.Ensemble].trainable = True
            model.config[nengo.Connection].trainable = True
            model.config[nengo.Connection].synapse = None

            self.input_node = nengo.Node(size_in=self.input_shape)
            layer_1 = nengo.Ensemble(n_neurons=self.nb_hidden,
                                     dimensions=self.input_shape,
                                     encoders=encoders,
                                     )
            layer_2 = nengo.Ensemble(n_neurons=self.nb_hidden,
                                     dimensions=self.input_shape
                                     )
            layer_3 = nengo.Ensemble(n_neurons=self.nb_hidden,
                                     dimensions=self.input_shape
                                     )
            output = nengo.Node(size_in=self.output_shape)

            conn_1 = nengo.Connection(self.input_node, layer_1)
            conn_2 = nengo.Connection(layer_1, layer_2)
            conn_3 = nengo.Connection(layer_2, layer_3)
            conn_4 = nengo.Connection(layer_3.neurons, output, transform=decoders.T)

            model.config[conn_1].trainable = True
            model.config[conn_2].trainable = True
            model.config[conn_3].trainable = True
            model.config[conn_4].trainable = True

            self.output_p = nengo.Probe(output)
        return model

    def training(self, input_data, label, batch_size, nb_epochs):

        with nengo_dl.Simulator(self.model, minibatch_size=batch_size,
                                step_blocks=1, device="/cpu:0", seed=2) as sim:

            sim.train({self.input_node: input_data},
                      {self.output_p: label},
                      tf.train.MomentumOptimizer(5e-2, 0.9),
                      #tf.train.GradientDescentOptimizer(learning_rate=0.05),
                      n_epochs=nb_epochs
                      )
            sim.save_params(self.weights_path)


    def predict(self, input_data, batch_size=1):

        with nengo_dl.Simulator(self.model, minibatch_size=batch_size,
                                step_blocks=1, device="/cpu:0", seed=1) as sim:

            sim.load_params(self.weights_path)
            sim.step(input_feeds={self.input_node: input_data})
            output = sim.data[self.output_p]
        return output

if __name__ == '__main__':

    from keras.datasets import mnist
    from keras.utils import np_utils
    from sklearn.metrics import accuracy_score

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # data pre-processing
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
    y_train = np_utils.to_categorical(y_train, nb_classes=10)
    y_test = np_utils.to_categorical(y_test, nb_classes=10)

    X_train_ = np.expand_dims(X_train, axis=1)
    X_test_ = np.expand_dims(X_test, axis=1)
    y_train_ = np.expand_dims(y_train, axis=1)
    y_test_ = np.expand_dims(y_test, axis=1)


    model = Spiking_Qnetwork(input_shape=28*28,
                             output_shape=10,
                             nb_hidden=1000,
                             weights_path="/home/huangbo/SpikingDeepRLControl/huangbo_ws/"
                                          "networks/saved_weights/snn_weights")

    # training
    model.training(input_data=X_train_, label=y_train_, batch_size=32, nb_epochs=10)

    output = model.predict(batch_size=X_test.shape[0], input_data=X_test_)
    prediction = np.squeeze(output, axis=1)

    # evaluate the model
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(prediction, axis=1))
    print "the test acc is:", acc