import numpy as np

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class Neuron:
    def __init__(self, num_inputs):
        # Initialize weights and bias randomly
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand(1)

    def feedforward(self, inputs):
        # Calculate weighted sum and apply activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron):
        # Create a list of neurons
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]

    def feedforward(self, inputs):
        # Get outputs from all neurons in the layer
        return np.array([neuron.feedforward(inputs) for neuron in self.neurons])

class NeuralNetwork:
    def __init__(self, layers):
        # layers is a list containing the number of neurons in each layer
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(Layer(layers[i + 1], layers[i]))

    def feedforward(self, inputs):
        # Propagate inputs through all layers
        for layer in self.layers:
            inputs = layer.feedforward(inputs)
        return inputs

    def train(self, training_inputs, training_outputs, learning_rate, epochs):
        for epoch in range(epochs):
            for inputs, expected_output in zip(training_inputs, training_outputs):
                # Forward pass
                activations = [inputs]
                for layer in self.layers:
                    activations.append(layer.feedforward(activations[-1]))

                # Backward pass
                error = expected_output - activations[-1]
                deltas = [error * sigmoid_derivative(activations[-1])]

                for i in reversed(range(len(self.layers) - 1)):
                    error = np.dot(self.layers[i + 1].neurons[0].weights, deltas[0])
                    deltas.insert(0, error * sigmoid_derivative(activations[i + 1]))

                # Update weights and biases
                for i, layer in enumerate(self.layers):
                    for j, neuron in enumerate(layer.neurons):
                        neuron.weights += learning_rate * deltas[i][j] * activations[i]
                        neuron.bias += learning_rate * deltas[i][j]

# Example usage:
if __name__ == "__main__":
    # Initialize neural network: 2 input neurons, 2 hidden neurons, 1 output neuron
    nn = NeuralNetwork([2, 2, 1])

    # XOR problem inputs and outputs
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_outputs = np.array([[0], [1], [1], [0]])

    # Train the neural network
    nn.train(training_inputs, training_outputs, learning_rate=0.5, epochs=10000)

    # Test the neural network
    for inputs in training_inputs:
        print(f"Input: {inputs} Output: {nn.feedforward(inputs)}")
