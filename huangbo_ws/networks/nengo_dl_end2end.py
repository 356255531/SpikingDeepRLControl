import nengo
import nengo_dl
import numpy as np
import tensorflow as tf

def cross_entropy(prediction, label):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=prediction))

with nengo.Network(seed=0) as model:

    nengo_dl.configure_trainable(model, default=True)

    model.config[nengo.Ensemble].neuron_type = nengo_dl.neurons.SoftLIFRate()
    model.config[nengo.Ensemble].intercepts=nengo.dists.Uniform(-1.0, 1.0)
    model.config[nengo.Ensemble].max_rates=nengo.dists.Choice([100])
    # model.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
    # model.config[nengo.Ensemble].bias = nengo.dists.Uniform(-1, 1)
    model.config[nengo.Ensemble].trainable = True
    model.config[nengo.Connection].trainable = True
    model.config[nengo.Connection].synapse = None

    # initialize encoder and decoder
    rng = np.random.RandomState(1)
    encoders = rng.normal(size=(1000, 784))
    decoders = rng.normal(size=(1000, 10))

    # network
    input_node = nengo.Node(size_in=784)
    layer_1 = nengo.Ensemble(n_neurons=1000,
                             dimensions=784,
                             encoders=encoders,
                             neuron_type=nengo_dl.neurons.SoftLIFRate()
                             )
    layer_2 = nengo.Ensemble(n_neurons=1000,
                             dimensions=784,
                             neuron_type=nengo_dl.neurons.SoftLIFRate()
                             )
    layer_3 = nengo.Ensemble(n_neurons=1000,
                             dimensions=784,
                             neuron_type=nengo_dl.neurons.SoftLIFRate()
                             )
    output = nengo.Node(size_in=10)

    conn_1 = nengo.Connection(input_node, layer_1)
    conn_2 = nengo.Connection(layer_1, layer_2)
    conn_3 = nengo.Connection(layer_2, layer_3)
    conn_4 = nengo.Connection(layer_3.neurons, output, transform=decoders.T)

    output_p = nengo.Probe(output)

with nengo_dl.Simulator(model, minibatch_size=60, step_blocks=1, device="/gpu:0", seed=2) as sim:

    from keras.datasets import mnist
    from keras.utils import np_utils


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

    sim.train({input_node: X_train_},
              {output_p: y_train_},
              #tf.train.MomentumOptimizer(5e-2, 0.9),
              tf.train.GradientDescentOptimizer(learning_rate=0.05),
              n_epochs=10,
              objective =cross_entropy
              )

    print "cross_entropy as loss is:", sim.loss({input_node: X_test_}, {output_p: y_test_}, cross_entropy)
    sim.save_params("/home/huangbo/SpikingDeepRLControl/huangbo_ws/networks/saved_weights/snn_weights")

with nengo_dl.Simulator(model, minibatch_size=10000, step_blocks=1, device="/gpu:0", seed=1) as sim:
    sim.load_params("/home/huangbo/SpikingDeepRLControl/huangbo_ws/networks/saved_weights/snn_weights")

    sim.step(input_feeds={input_node: X_test_})
    output = sim.data[output_p]
    prediction = np.squeeze(output, axis=1)

    # evaluate the model
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(prediction, axis=1))
    print "the test acc is:", acc



    # sim.step(input_feeds={input_node: np.expand_dims(X_test_[123, :, :], axis=0)})
    # output = sim.data[output_p]
    # output = np.squeeze(output, axis=1)
    # print output
    # print np.argmax(output)
    #
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(X_test[123,:].reshape(28,28))
    # plt.show()
