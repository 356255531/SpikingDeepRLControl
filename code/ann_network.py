from keras.layers import Dense
from keras.models import Sequential
from keras.backend import set_image_dim_ordering


class Q_learning_network(object):

    set_image_dim_ordering("tf")

    def __init__(self, input_shape, output_shape, weights_file):
        '''
        The Q_value function approximation based on artificial neural network
        :param input_shape: dimension of input
        :param nb_classes: dimension of output
        :param weights_file: path to save the weights
        '''
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights_file = weights_file

        self.network = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(100, input_dim=self.input_shape, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.output_shape, activation='linear'))
        return model

    def run_training(self, training_data, label):
        self.network.compile(optimizer="rmsprop",
                             loss='mse',
                             metrics=['accuracy'],
                             )
        self.network.train_on_batch(x=training_data, y=label)

    def prediction(self, input):
        output = self.network.predict(input)
        return output

    def save_weights(self):
        self.network.save_weights(self.weights_file)

    def load_weights(self):
        self.network.load_weights(self.weights_file)


