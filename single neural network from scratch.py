import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    A simple single-layer neural network with sigmoid activation function.
    """

    def __init__(self, input_size, output_size, learning_rate=0.01):
        """
        Initializes the neural network with random weights and biases.

        Args:
            input_size (int): Number of features in the input data.
            output_size (int): Number of output neurons (typically 1 for regression).
            learning_rate (float, optional): Learning rate for gradient descent. Defaults to 0.01.
        """
        self.weights = np.random.rand(input_size, output_size)  # Random weights
        self.bias = np.random.rand(output_size)  # Random bias
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output data after applying sigmoid activation.
        """
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        """
        Predicts the output of the network for a given input.

        Args:
            X (numpy.ndarray): Input data (single sample or batch).

        Returns:
            numpy.ndarray: Predicted output.
        """
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def train(self, X, y, epochs=100):
        """
        Trains the network using gradient descent.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target labels.
            epochs (int, optional): Number of training epochs. Defaults to 100.
        """
        for _ in range(epochs):
            # Forward propagation
            y_pred = self.predict(X)

            # Calculate loss (mean squared error)
            loss = np.mean((y - y_pred) ** 2)

            # Backpropagation (calculate gradients)
            d_loss_d_output = 2 * (y - y_pred)  # Derivative of loss w.r.t. output
            d_output_d_z = self.sigmoid_derivative(y_pred)  # Derivative of output w.r.t. weighted sum (z)
            z = np.dot(X, self.weights) + self.bias  # Weighted sum
            d_z_d_weights = X  # Derivative of z w.r.t. weights
            d_z_d_bias = np.ones((X.shape[0], 1))  # Derivative of z w.r.t. bias

            # Update weights and biases
            self.weights -= self.learning_rate * np.dot(d_z_d_weights.T, d_loss_d_output * d_output_d_z)
            self.bias -= self.learning_rate * np.sum(d_loss_d_output * d_output_d_z, axis=0)

            # Print loss for every 10 epochs
            if (_ % 10) == 0:
                print(f"Epoch {_}: Loss = {loss:.4f}")

    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid activation function.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Derivative of sigmoid activation.
        """
        return self.sigmoid(x) * (1 - self.sigmoid(x))

# Example: Linear regression with a single neuron

# Generate sample data
X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
y = 2 * X + 1  # Target linear relationship

# Create and train the neural network
model = NeuralNetwork(input_size=1, output_size=1)
model.train(X.reshape(-1, 1), y.reshape(-1, 1), epochs=1000)

# Generate data for visualization
X_vis = np.linspace(0, 0.5, 100)

# Predict output for visualization
