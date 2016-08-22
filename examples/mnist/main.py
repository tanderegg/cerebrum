from load import transform
from cerebrum.neural_network import NeuralNetwork

if __name__ == "__main__":
    training_data, validation_data, test_data = transform()
    net = NeuralNetwork([784, 100, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
