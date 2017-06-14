import nengo
import nengo_dl
import numpy as np


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
            input_neuron = nengo.Ensemble(n_neurons=self.nb_hidden,
                                          dimensions=self.input_shape,
                                          neuron_type=nengo.LIFRate(),
                                          intercepts=nengo.dists.Uniform(-1.0, 1.0),
                                          max_rates=nengo.dists.Choice([100]),
                                          encoders=encoders,
                                          )
            output = nengo.Node(size_in=self.output_shape)
            nengo.Connection(input_neuron,
                             output,
                             synapse=None,
                             transform=decoders.T
                             )
        return model
    


