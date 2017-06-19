import nengo
import numpy as np


class DQN:
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
