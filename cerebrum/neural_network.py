### NeuralNetwork class, adapted from
### http://neuralnetworksanddeeplearning.com/chap1.html
import random

import numpy

### Example algorithms

def random_init(layers):
    """
    An example initializer function which generates initial weights and biases
    with a Gaussian distribution and stddev of 1.  Note that 'biases' is one
    dimension less than layer_sizes, as the input layer doesn't use biases.
    """
    biases = [numpy.random.randn(y) for y in layers[1:]]
    weights = [numpy.random.randn(y, x)
        for x, y in zip(layers[:-1], layers[1:])]
    return biases, weights


def sigmoid(z):
    """
    Runs the Sigmoid algorithm on 'z'.  'z' can be either a scalar, vector,
    or any other numpy 'array_like' type.
    """
    return 1.0/(1.0+numpy.exp(-z))


def sigmoid_prime(z):
    """
    Derivative of the sigmoid function.
    """
    return sigmoid(z)*(1-sigmoid(z))


class NeuralNetwork(object):

    def __init__(self, layers,
                 initializer=random_init,
                 initializer_params=dict()):
        """
        layers is a list of sizes for each layer in the Neural Network. The
        first layer is always the input layer, the last the output layer. A
        value of [5, 7, 3], for example, would indicate 5 input neurons, a 7
        neuron hidden layer, and 3 output neurons. 'initializer' is the
        initialization function, which should take the layers list and any
        optional keyword arguments.
        """
        self.num_layers = len(layers)
        self.layers = layers
        self.biases, self.weights = initializer(
            layers, **initializer_params
        )


    def feedforward(self, inputs, f=sigmoid):
        """
        Return the output of the neural net on 'inputs'.  The weights are
        combined with the inputs via dot product, and the biases added to the
        result, which is passed to f (the sigmoid function by default).
        """
        for biases, weights in zip(self.biases, self.weights):
            inputs = f(numpy.dot(weights, inputs) + biases)
        return inputs


    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,
            test_data=None):
        """
        Stocastic Gradient Decent is used to estimate the mimimum of the cost
        function, thus improving the neural net's model.  'training_data' is
        a list of tuples (x, y) representing each input and desired outputs.
        Providing some 'test_data' causes the network to be evaluated after
        each epoch.
        """
        if test_data: test_data_length = len(test_data)
        training_data_length = len(training_data)

        for j in list(range(epochs)):
            random.shuffle(training_data)

            # Generate minibatches after randomizing the training data
            mini_batches = [
                training_data[k:k+mini_batch_size]
                    for k in list(range(0, training_data_length, mini_batch_size))
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), test_data_length
                ))
            else:
                print("Epoch {0} complete.".format(j))


    def update_mini_batch(self, mini_batch, learning_rate):
        """
        Update the network's weights and biases by applying gradient descent
        using backpropagation to a single minibatch.
        """
        gradients_b = [numpy.zeros(biases.shape) for biases in self.biases]
        gradients_w = [numpy.zeros(weights.shape) for weights in self.weights]

        for x_in, y_out in mini_batch:
            delta_gradients_b, delta_gradients_w = self.backprop(x_in, y_out)
            gradients_b = [
                gb+dgb for gb, dgb in zip(
                    gradients_b, delta_gradients_b
                )
            ]
            gradients_w = [
                gw+dgw for gw, dgw in zip(
                    gradients_w, delta_gradients_w
                )
            ]

        self.weights = [
            w-(learning_rate/len(mini_batch))*nw for w, nw in zip(
                self.weights, gradients_w
            )
        ]
        self.biases = [
            b-(learning_rate/len(mini_batch))*nb for b, nb in zip(
                self.biases, gradients_b
            )
        ]


    def backprop(self, x_in, y_out):
        """
        Backpropegation first takes the input values, applies the weights and
        biases, then runs the activation function on each result, for each
        layer. Next, it figures out the value of the cost_derivative function
        on the final layer of activations compared to the desired output times
        the derivative of the activation function, and sets the gradients
        accordingly.  Finally, it back propegates those gradients through the
        network to generate the gradients for all other layers.
        """
        gradients_b = [numpy.zeros(biases.shape) for biases in self.biases]
        gradients_w = [numpy.zeros(weights.shape) for weights in self.weights]

        # Run the activation on all biases and weights based on the input
        activation = x_in
        activations = [x_in]

        weighted_inputs = []
        for biases, weights in zip(self.biases, self.weights):
            weighted_input = numpy.dot(weights, activation) + biases
            weighted_inputs.append(weighted_input)
            activation=sigmoid(weighted_input)
            activations.append(activation)

        # Now, calculate the cost function change on the final layer and
        # update the gradients of that layer.
        delta = self.cost_derivative(activations[-1], y_out) * \
            sigmoid_prime(weighted_inputs[-1])
        gradients_b[-1] = delta
        gradients_w[-1] = numpy.dot(delta, activations[-1].transpose())

        # Continue propagate backwards to do the rest
        for layer in list(range(2, self.num_layers)):
            weighted_input = weighted_inputs[-layer]
            sp = sigmoid_prime(weighted_input)
            delta = numpy.dot(self.weights[-layer+1].transpose(), delta) * sp
            gradients_b[-layer] = delta
            gradients_w[-layer] = numpy.dot(
                delta, activations[-layer].transpose()
            )

        return gradients_b, gradients_w


    def cost_derivative(self, activation, y_out):
        """
        The cost derivative, in this case a simple difference between the
        desired outputs and the result of this round of activations.
        """
        return (activation-y_out)


    def evaluate(self, inputs):
        """
        Takes a series of inputs, and returns the indexes of the output chosen
        as the correct one for each.
        """
        test_results = [numpy.argmax(self.feedforward(x_in))
            for (x_in, y_out) in inputs]
        return test_results
