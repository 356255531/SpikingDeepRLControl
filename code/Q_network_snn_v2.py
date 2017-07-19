import nengo
import numpy as np


class LoadFrom(nengo.solvers.Solver):
    def __init__(self, filename, weights=False):
        super(LoadFrom, self).__init__(weights=weights)
        self.filename = filename

    def __call__(self, A, Y, rng=None, E=None):
        if self.weights:
            shape = (A.shape[1], E.shape[1])
        else:
            shape = (A.shape[1], Y.shape[1])
        try:
            value = np.load(self.filename)
            assert value.shape == shape
        except IOError:
            value = np.zeros(shape)
        return value, {}


# helper to create the LoadFrom solver and the needed probe and do the saving
class WeightSaver(object):
    def __init__(self, connection, filename, sample_every=1.0, weights=False):
        assert isinstance(connection.pre, nengo.Ensemble)
        if not filename.endswith('.npy'):
            filename = filename + '.npy'
        self.filename = filename
        connection.solver = LoadFrom(self.filename, weights=weights)
        self.probe = nengo.Probe(connection, 'weights', sample_every=sample_every)
        self.connection = connection

    def save(self, sim):
        np.save(self.filename, sim.data[self.probe][-1].T)
        print "weights saved"


class Q_network:

    def __init__(self, input_shape, output_shape, nb_hidden, encoder, decoder):
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
        self.encoder = encoder
        self.decoder = decoder

    def encoder_initialization(self):
        '''
        encoder is the connection relationship between input and the ensemble
        :return: initialised encoder
        '''

        rng = np.random.RandomState(1)
        encoders = rng.normal(size=(self.nb_hidden, self.input_shape))
        return encoders

    def train_network(self, train_data, train_targets):
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
                                          intercepts=nengo.dists.Uniform(-1.0, 1.0),
                                          max_rates=nengo.dists.Choice([100]),
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
            #encoders_weights = nengo.Probe(input_neuron.encoders, "encoders_weights", sample_every=1.0)
            #conn_weights = nengo.Probe(conn, 'weights', sample_every=1.0)
            with nengo.Simulator(model) as sim:
                sim.run(3)
                ws = WeightSaver(conn, 'my_weights')  # add this line
                ws.save(sim)


        # with nengo.Simulator(model) as sim:
        #     sim.run(3)
        # #save the connection weights after training
        # np.save(self.encoder, sim.data[input_neuron].encoders)
        # np.save(self.decoder, sim.data[conn_weights][-1].T)

    def predict(self, input):
        '''
        prediction after training, the output will be the corresponding q values
        :param input: input must be a numpy array, system state and action paars, shape = (dim_sample)
        :return: the q values
        '''

        encoders = self.encoder_initialization()
        decoder = np.load('/home/huangbo/SpikingDeepRLControl/code/my_weights.npy')
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
            conn = nengo.Connection(input_neuron.neurons,
                                    output,
                                    synapse=None,
                                    transform=decoder.T
                                    )
        with nengo.Simulator(model) as sim:
            _, acts = nengo.utils.ensemble.tuning_curves(input_neuron, sim, inputs=input)
        return np.dot(acts, sim.data[conn].weights.T)


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
                      encoder="/home/huangbo/Desktop/encoder.npy",
                      decoder="/home/huangbo/Desktop/decoder.npy")

    # training
    model.train_network(X_train, y_train)
    prediction = model.predict(X_test)
    #print prediction.shape

    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(prediction, axis=1))
    print "the test acc is:", acc
