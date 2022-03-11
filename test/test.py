# BMI 203 Project 7: Neural Network

import nn
import numpy as np

# TODO: Write your test functions and associated docstrings below.

# instantiate a NeuralNet
net = nn.NeuralNetwork([{'input_dim': 68, 'output_dim': 34, 'activation': "sigmoid"},
                                        {'input_dim': 34, 'output_dim': 17, 'activation': "relu"},
                                        {"input_dim": 17, 'output_dim': 1, 'activation': "sigmoid"}],
                                    0.1, 42, 10, 10, "mse")
# generate fake data for testing
np.random.seed(42)
train = np.random.random((500, 69))
X_train, y_train = train[:, 0:68], train[:, -1]


def test_forward():
    A_curr, cache = net.forward(X_train)
    assert len(cache) == 2*len(net.arch) + 1 # check `cache` is correct length
    assert np.all(cache["A0"] == X_train) # check first element of cache is input data

    # check that each `A` layer of cache is the corresponding `Z` layer passed through an activation
    for layer in range(1,len(net.arch)+1):
        print(layer)
        if net.arch[layer-1]["activation"] == "sigmoid":
            assert np.all(cache[f"A{layer}"] == net._sigmoid(cache[f"Z{layer}"]))
        else:
            assert np.all(cache[f"A{layer}"] == net._relu(cache[f"Z{layer}"]))

def test_single_forward():
    pass


def test_single_backprop():
    pass


def test_predict():
    pass


def test_binary_cross_entropy():
    pass


def test_binary_cross_entropy_backprop():
    pass


def test_mean_squared_error():
    pass


def test_mean_squared_error_backprop():
    pass


def test_one_hot_encode():
    pass


def test_sample_seqs():
    pass
