# BMI 203 Project 7: Neural Network

import nn
import numpy as np
import pytest

# TODO: Write your test functions and associated docstrings below.

# instantiate a NeuralNet
net = nn.NeuralNetwork([{'input_dim': 68, 'output_dim': 34, 'activation': "sigmoid"},
                                        {'input_dim': 34, 'output_dim': 17, 'activation': "relu"},
                                        {"input_dim": 17, 'output_dim': 1, 'activation': "sigmoid"}],
                                    0.1, 42, 10, 10, "mse")
# generate fake data for testing
np.random.seed(42)
train = np.random.random((500, 69))
X_train, y_train = train[:, 0:68], train[:, -1].reshape(500, 1)
train = np.random.random((500, 69))
X_val, y_val = train[:, 0:68], train[:, -1].reshape(500,1)


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
    A_curr, cache = net.forward(X_train)

    # check `Z` and `A` are being computed correctly
    for layer in range(1, len(net.arch)+1):
        assert np.all(cache[f"Z{layer}"] == cache[f"A{layer-1}"].dot(net._param_dict[f"W{layer}"].T)
                      + net._param_dict[f"b{layer}"].T)

def test_single_backprop():
    A_curr, cache = net.forward(X_train)

    net.fit(X_train, y_train, X_val, y_val)
    y_pred = net.predict(X_train)

    dA_curr = net._mean_squared_error_backprop(y_train, y_pred)

    for layer in range(len(net.arch), 0, -1):
        dA_prev, dW_curr, db_curr = net._single_backprop(net._param_dict[f"W{layer}"],
                                                          net._param_dict[f"b{layer}"],
                                                          cache[f"Z{layer}"],
                                                          cache[f"A{layer - 1}"],
                                                          dA_curr,
                                                          net.arch[layer - 1]["activation"])

        if net.arch[layer-1]["activation"] == "relu":
            dZ_curr = net._relu_backprop(cache[f"Z{layer}"])
        else:
            dZ_curr = net._sigmoid_backprop(cache[f"Z{layer}"])

        # check these are being updated correctly
        assert np.all(dA_prev == (dA_curr*dZ_curr).dot(net._param_dict[f"W{layer}"]))
        assert np.all(dW_curr == (cache[f"A{layer - 1}"].T).dot(dA_curr*dZ_curr).T)
        assert np.all(db_curr == np.sum((dA_curr*dZ_curr), axis=0).reshape(net._param_dict[f"b{layer}"].shape))

def test_predict():
    net = nn.NeuralNetwork([{'input_dim': 68, 'output_dim': 34, 'activation': "sigmoid"},
                            {'input_dim': 34, 'output_dim': 17, 'activation': "relu"},
                            {"input_dim": 17, 'output_dim': 1, 'activation': "sigmoid"}],
                           0.1, 42, 10, 10, "mse")
    with pytest.raises(ValueError):
        net.predict(X_train)

def test_binary_cross_entropy():
    assert np.isclose(net._binary_cross_entropy(np.array([0, 0, 1, 1]), np.array([0.1, 0.3, 0.5, 0.7])), 0.3779643)


def test_binary_cross_entropy_backprop():
    assert np.isclose(np.sum(net._binary_cross_entropy_backprop(np.array([0, 0, 1, 1]), np.array([0.1, 0.3, 0.5, 0.7]))
                             - np.array([0.27777778, 0.35714286, -0.5, -0.35714286])), 0)

def test_mean_squared_error():
    assert np.isclose(net._mean_squared_error(np.array([0.27163318, 0.19060983, 0.18463246, 0.34210762]),
                                              np.array([[0.42995762, 0.83094917, 0.11989109, 0.73492585]])), 0.14839967)


def test_mean_squared_error_backprop():
    assert np.isclose(np.sum(net._mean_squared_error_backprop(np.array([0.27163318, 0.19060983, 0.18463246, 0.34210762]),
                                              np.array([[0.42995762, 0.83094917, 0.11989109, 0.73492585]]))-
                             np.array([[ 0.07916222, 0.32016967, -0.03237068, 0.19640911]])), 0)

def test_one_hot_encode():
    assert np.all(np.array(nn.preprocess.one_hot_encode_seqs("ATCG")) == np.eye(4))
    with pytest.raises(ValueError):
        nn.preprocess.one_hot_encode_seqs("ACTGR")


def test_sample_seqs():
    data = np.random.random((1000, 50))
    labels = [0 for x in range(900)] + [1 for x in range(100)]
    d, l = nn.preprocess.sample_seqs(data, labels)
    assert np.abs(np.mean(l) - 0.5) < 0.05

