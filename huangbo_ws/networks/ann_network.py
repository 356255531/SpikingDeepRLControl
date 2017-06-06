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

    def train_on_batch(self, training_data, label):
        self.network.compile(optimizer="rmsprop",
                             loss='mse',
                             metrics=['accuracy'],
                             )
        loss = self.network.train_on_batch(x=training_data, y=label)
        print "loss", loss

    def prediction(self, input):
        output = self.network.predict(input)
        return output

    def save_weights(self):
        self.network.save_weights(self.weights_file)

    def load_weights(self):
        self.network.load_weights(self.weights_file)


if __name__ == '__main__':

    from keras.datasets import mnist
    from keras.utils import np_utils
    import matplotlib.pyplot as plt
    import numpy as np

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # data pre-processing
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.  # normalize
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.  # normalize
    y_train = np_utils.to_categorical(y_train, nb_classes=10)
    y_test = np_utils.to_categorical(y_test, nb_classes=10)

    def batch_generator(batch_size):
        index = 0
        while(1):
            image_batch = X_train[index:index+batch_size,:]
            label_batch = y_train[index:index+batch_size,:]

            index = index + batch_size
            yield image_batch, label_batch


    Q_net = Q_learning_network(input_shape=28*28, output_shape=10, weights_file="weights.hdf5")

    for i in range(50):
        image_batch, label_batch = batch_generator(32).next()
        Q_net.train_on_batch(image_batch, label_batch)

    test_image = X_test[123,:]
    image = test_image.reshape(28,28)
    test_image = np.squeeze(test_image)

    plt.figure()
    plt.imshow(image)
    plt.show()
    test_image = test_image.reshape(1,784)

    print Q_net.prediction(test_image)

    print np.argmax(Q_net.prediction(test_image))






