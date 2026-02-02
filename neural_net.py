import numpy as numpy

X = numpy.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

Y = numpy.array([
    [0],
    [1],
    [1],
    [0]
])

numpy.random.seed(42)

weights_input_hidden = numpy.random.randn(2,4)
bias_hidden = numpy.zeros((1,4))

weights_hidden_output = numpy.random.randn(4,1)
bias_output = numpy.zeros((1,1))

print("Training data (X):")
print(X)

print("\nTarget output (Y):")
print(Y)

print("\nInitial weights from input to hidden layer:")
print(weights_input_hidden)

print("\nInitial biases for hidden layer:")
print(bias_hidden)

print("\nInitial weights from hidden to output layer:")
print(weights_hidden_output)

print("\nInitial biases for output layer:")
print(bias_output)





