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

