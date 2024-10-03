import numpy as np

class DeepNeuralNetwork:
    def __init__(self, layer_sizes):
        self.params = self.initialize_parameters(layer_sizes)

    def initialize_parameters(self, layer_sizes):
        np.random.seed(1)
        parameters = {}
        L = len(layer_sizes)  # number of layers
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_sizes[l], 1))
        return parameters

    def forward_propagation(self, X):
        cache = {'A0': X}
        L = len(self.params) // 2  # number of layers
        A = X
        for l in range(1, L + 1):
            Z = np.dot(self.params['W' + str(l)], A) + self.params['b' + str(l)]
            A = ActivationFunction.relu(Z) if l < L else ActivationFunction.softmax(Z)
            cache['A' + str(l)] = A
            cache['Z' + str(l)] = Z
        return A, cache

    def compute_loss(self, AL, Y):
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(AL + 1e-8)) / m
        return np.squeeze(loss)

    def backward_propagation(self, Y, cache):
        grads = {}
        L = len(self.params) // 2  # number of layers
        m = Y.shape[1]

        # Initialize backpropagation from output layer
        dZL = cache['A' + str(L)] - Y
        grads['dW' + str(L)] = (1 / m) * np.dot(dZL, cache['A' + str(L - 1)].T)
        grads['db' + str(L)] = (1 / m) * np.sum(dZL, axis=1, keepdims=True)

        # Backpropagation through hidden layers
        for l in reversed(range(1, L)):
            dZ = np.dot(self.params['W' + str(l + 1)].T, dZL) * self.relu_derivative(cache['Z' + str(l)])
            grads['dW' + str(l)] = (1 / m) * np.dot(dZ, cache['A' + str(l - 1)].T)
            grads['db' + str(l)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dZL = dZ  # propagate error backward

        return grads

    def update_parameters(self, grads, learning_rate):
        L = len(self.params) // 2  # number of layers
        for l in range(1, L + 1):
            self.params['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
            self.params['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    @staticmethod
    def relu_derivative(Z):
        return np.where(Z > 0, 1, 0)

# Activation Functions class for reuse
class ActivationFunction:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

