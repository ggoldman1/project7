# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike


# Neural Network Class Definition
class NeuralNetwork:
    """
    This is a neural network class that generates a fully connected Neural Network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            This list of dictionaries describes the fully connected layers of the artificial neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation': 'sigmoid'}] will generate a
            2 layer deep fully connected network with an input dimension of 64, a 32 dimension hidden layer
            and an 8 dimensional output.
        lr: float
            Learning Rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            This list of dictionaries describing the fully connected layers of the artificial neural network.
    """
    def __init__(self,
                 nn_arch: List[Dict[str, Union[int, str]]],
                 lr: float,
                 seed: int,
                 batch_size: int,
                 epochs: int,
                 loss_function: str):
        # Saving architecture
        self.arch = nn_arch
        # Saving hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size
        # Initializing the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD!! IT IS ALREADY COMPLETE!!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """
        # seeding numpy random
        np.random.seed(self._seed)
        # defining parameter dictionary
        param_dict = {}
        # initializing all layers in the NN
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            # initializing weight matrices
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            # initializing bias matrices
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
        return param_dict

    def _single_forward(self,
                        W_curr: ArrayLike,
                        b_curr: ArrayLike,
                        A_prev: ArrayLike,
                        activation: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        Z_curr = A_prev.dot(W_curr.T) + b_curr.T
        if activation == "relu":
            A_curr = self._relu(Z_curr)
        else:
            A_curr = self._sigmoid(Z_curr)
        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        cache = {}
        cache["A0"] = X
        A_prev = X
        for layer in range(1, len(self.arch) + 1):
            A_curr, Z_curr = self._single_forward(self._param_dict[f"W{layer}"],
                                                  self._param_dict[f"b{layer}"],
                                                  A_prev, self.arch[layer-1]["activation"]) # pass through one layer
            cache[f"A{layer}"] = A_curr # store A
            cache[f"Z{layer}"] = Z_curr # store Z
            A_prev = A_curr # update prev pointer
        return A_curr, cache

    def _single_backprop(self,
                         W_curr: ArrayLike,
                         b_curr: ArrayLike,
                         Z_curr: ArrayLike,
                         A_prev: ArrayLike,
                         dA_curr: ArrayLike,
                         activation_curr: str) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        if activation_curr == "relu":
            dZ_curr = self._relu_backprop(Z_curr)
        else:
            dZ_curr = self._sigmoid_backprop(Z_curr)

        dA_prev = (dA_curr*dZ_curr).dot(W_curr)
        dW_curr = (A_prev.T).dot(dA_curr*dZ_curr).T # make sure this aligns with dimensions of self._param_dict[Wx]
        db_curr = np.sum((dA_curr*dZ_curr), axis=0).reshape(b_curr.shape) # make sure this is the same dimension as bias
        
        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        grad_dict = {}

        if self._loss_func == 'mse':
            dA_curr = self._mean_squared_error_backprop(y, y_hat)
        else:
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)

        for layer in range(len(self.arch), 0, -1):

            dA_prev, dW_curr, db_curr = self._single_backprop(self._param_dict[f"W{layer}"],
                                                              self._param_dict[f"b{layer}"],
                                                              cache[f"Z{layer}"],
                                                              cache[f"A{layer-1}"],
                                                              dA_curr,
                                                              self.arch[layer-1]["activation"])

            grad_dict[f"dA_prev{layer}"] = dA_prev
            grad_dict[f"dW_curr{layer}"] = dW_curr
            grad_dict[f"db_curr{layer}"] = db_curr
            dA_curr = dA_prev

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and thus does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.

        Returns:
            None
        """
        for layer in range(1, len(self.arch)):
            self._param_dict[f"W{layer}"] = self._param_dict[f"W{layer}"] - self._lr*grad_dict[f"dW_curr{layer}"]
            self._param_dict[f"b{layer}"] = self._param_dict[f"b{layer}"] - self._lr * grad_dict[f"db_curr{layer}"]

    def fit(self,
            X_train: ArrayLike,
            y_train: ArrayLike,
            X_val: ArrayLike,
            y_val: ArrayLike) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network via training for the number of epochs defined at
        the initialization of this class instance.
        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        epoch = 1
        param_update = 1

        target_dim = self.arch[-1]["output_dim"]

        while param_update > 0.001 and epoch < self._epochs:
            param_update = 0

            # shuffle the training data
            shuf = np.concatenate([X_train, y_train], axis=1)
            np.random.shuffle(shuf)
            X_train = shuf[:, :target_dim]
            y_train = shuf[:,target_dim:].reshape(len(y_train), target_dim)

            # get the training batches
            num_batches = int(X_train.shape[0]/self._batch_size) + 1
            X_batch = np.array_split(X_train, num_batches)
            y_batch = np.array_split(y_train, num_batches)

            within_epoch_loss_train = []
            within_epoch_loss_val = []
            for X, y in zip(X_batch, y_batch): # one epoch

                y_hat, cache = self.forward(X)

                # update parameters via backprop
                grad_dict = self.backprop(y, y_hat, cache)
                old_params = self._param_dict.copy()
                self._update_params(grad_dict)
                param_update += self._get_param_update_magnitude(old_params, self._param_dict)

                # keep track of validation and training loss
                y_hat_val = self.predict(X_val)
                if self._loss_func == "mse":
                    loss_train = self._mean_squared_error(y, y_hat)
                    loss_val = self._mean_squared_error(y_val, y_hat_val)
                else:
                    loss_train = self._binary_cross_entropy(y_train, y_hat)
                    loss_val = self._binary_cross_entropy(y_val, y_hat_val)
                within_epoch_loss_train.append(loss_train)
                within_epoch_loss_val.append(loss_val)

            param_update /= num_batches # average param update magnitude across the # of batches

            per_epoch_loss_train.append(np.mean(within_epoch_loss_train))
            per_epoch_loss_val.append(np.mean(within_epoch_loss_train))


            epoch += 1

        return per_epoch_loss_train, per_epoch_loss_val


    def _get_param_update_magnitude(self, old_params: ArrayLike, new_params: ArrayLike) -> float:
        """
        Get the average absolute value parameter difference after updating with backpropagation.

        Args:
            old_params: ArrayLike
                Previous parameters
            new_params: ArrayLike
                New parameters

        Return:
            float
                Average difference in old vs new parameters
        """
        avg_update = []
        for param in old_params.keys():
            avg_update.append(np.mean(np.abs(old_params[param] - new_params[param])))
        return np.mean(avg_update)

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network model.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        return self.forward(X)[0]

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return 1/(1+np.exp(Z))

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0, Z)

    def _sigmoid_backprop(self, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = self._sigmoid(Z)*(1-self._sigmoid(Z))
        return dZ

    def _relu_backprop(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ = np.where(Z > 0, Z, 0)
        return dZ


    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        return -((y * np.log(y_hat)) + ((1-y) * (np.log(1-y_hat)))).mean()

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        return -((y/yhat) - ((1-y)/(1-yhat))) / len(y)

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        return np.square(y-y_hat).mean()

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        return 2*(y_hat - y) / len(y)

    def _loss_function(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Loss function, computes loss given y_hat and y. This function is
        here for the case where someone would want to write more loss
        functions than just binary cross entropy.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """
        pass

    def _loss_function_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        This function performs the derivative of the loss function with respect
        to the loss itself.
        Args:
            y (array-like): Ground truth output.
            y_hat (array-like): Predicted output.
        Returns:
            dA (array-like): partial derivative of loss with respect
                to A matrix.
        """
        pass

# num_data = 100
#
# nn = NeuralNetwork([{'input_dim': 10, 'output_dim': 5, 'activation': 'relu'},
#                     {'input_dim': 5, 'output_dim': 1, 'activation': 'sigmoid'}], .1, 42, 10, 10, "mse")
# data = np.random.random((num_data, 10))
# train, val = np.array_split(data, 2)
# target_train, target_val = np.array_split(np.random.random((num_data,1)), 2)
# # output, cache = nn.forward(data)
# # grad_dict = nn.backprop(target, output, cache)
# train_loss, val_loss = nn.fit(train, target_train, val, target_val)

from sklearn.datasets import load_digits
data = load_digits()["data"]
n = data.shape[0]
train_rows = int(0.8*n) # 80% train, 20% test
np.random.shuffle(data)
X_train, X_val = data[0:train_rows, :], data[train_rows:n, :]

net = NeuralNetwork([{'input_dim': 64, 'output_dim': 16, 'activation': "relu"},
                                     {'input_dim': 16, 'output_dim': 64, 'activation': "relu"}],
                                    1, 42, 10, 10, "mse")

net.fit(X_train, X_train, X_val, X_val)