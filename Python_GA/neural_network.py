import numpy as np

# TODO make dense layer object
class Dense_layer:
    def __init__(self, number_of_inputs, number_of_neurons, activation_function):
        self.weights = np.random.uniform(-1, 1, (number_of_inputs, number_of_neurons))
        self.biases = np.zeros((1, number_of_neurons))
        self.activation = activation_function

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        if self.activation == "ReLu":
            self.activation_ReLu()
        if self.activation == "softmax":
            self.activation_softmax()
        if self.activation == "sigmoid":
            self.activation_sigmoid()
    
    def activation_softmax(self):
        # Get unnormalized probabilities
        exp_values = np.exp(self.output - np.max(self.output, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def activation_ReLu(self):
        self.output = np.maximum(0, self.output)

    def activation_sigmoid(self):
        self.output = 1 / (1 + np.exp(-self.output))

# TODO make feed forward network
class Feedforward_neural_network:
    def __init__(self, number_of_inputs, number_of_outputs, number_of_hidden_layers, number_of_neurons_per_hidden_layer):
        self.layers = []

        # Add input layer
        self.layers.append(Dense_layer(number_of_inputs, number_of_neurons_per_hidden_layer, "sigmoid"))

        # Add hidden layers
        for i in range(number_of_hidden_layers):
            self.layers.append(Dense_layer(number_of_neurons_per_hidden_layer, number_of_neurons_per_hidden_layer, "sigmoid"))

        # Add output layer
        self.layers.append(Dense_layer(number_of_neurons_per_hidden_layer, number_of_outputs, "softmax"))

    def forward(self, inputs):
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.output
        return inputs