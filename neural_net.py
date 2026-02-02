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

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def forward(X):
    hidden = sigmoid(numpy.dot(X, weights_input_hidden) + bias_hidden)
    output = sigmoid(numpy.dot(hidden, weights_hidden_output) + bias_output)
    return hidden, output

hidden, output = forward(X)

def calculate_loss(y_true, y_pred):
    return numpy.mean((y_true - y_pred) ** 2)

def sigmoid_derivative(x):
    return x * (1 - x)

def backward(X, Y, hidden, output, learning_rate=0.5):
    global weights_input_hidden, bias_hidden
    global weights_hidden_output, bias_output

    # How wrong were we at the output? 
    output_error = Y - output
    output_delta = output_error * sigmoid_derivative(output)

    # How much did each hidden neuron contribute to the output error (according to the weights)?
    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden)

    # Update weights and biases
    weights_hidden_output += hidden.T.dot(output_delta) * learning_rate
    bias_output += numpy.sum(output_delta, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
    bias_hidden += numpy.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

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

print("\nOutput from the hidden layer before training:")
print(hidden)

print("\nOutput from the network before training:")
print(output)

print("\nInitial loss before training:")
print(calculate_loss(Y, output))

# Training loop
for epoch in range(10000):
    hidden, output = forward(X)
    backward(X, Y, hidden, output)

    if epoch % 1000 == 0:
        loss = calculate_loss(Y, output)
        print(f"Epoch {epoch}, Loss: {loss}")

print("\nFinal weights from input to hidden layer after training:")
print(weights_input_hidden)

print("\nFinal biases for hidden layer after training:")
print(bias_hidden)

print("\nFinal weights from hidden to output layer after training:")
print(weights_hidden_output)

print("\nFinal biases for output layer after training:")
print(bias_output)

print("\nOutput from the hidden layer after training:")
hidden, output = forward(X)
print(hidden)
print("\nOutput from the network after training:")
print(output)   

print("\nFinal loss after training:")
print(calculate_loss(Y, output))    

