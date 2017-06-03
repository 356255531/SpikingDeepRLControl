import nengo
import numpy as np
from vision import Gabor, Mask


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

    def train_network(self, train_data, train_targets, simulation_time=100):
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
            sim.run(simulation_time)

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
        with nengo.Simulator(model) as sim:
            _, acts = nengo.utils.ensemble.tuning_curves(input_neuron, sim, inputs=input)

        return np.dot(acts, sim.data[conn].weights.T)

