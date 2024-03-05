import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    """Derivative of the sigmoid activation function."""
    return x * (1 - x)

def backpropagation(x, y, w1, w2, b1, b2, learning_rate):
    """
    Performs backpropagation algorithm to update weights and biases.

    Args:
        x: Input data (1x2 array).
        y: Target output (1x1 array).
        w1: Weight between input layer and first hidden neuron.
        w2: Weight between first hidden neuron and output neuron.
        b1: Bias of the first hidden neuron.
        b2: Bias of the output neuron.
        learning_rate: Learning rate for weight and bias updates.

    Returns:
        Updated weights and biases.
    """
    # Forward pass
    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

    # Calculate error
    error = y - a2

    # Backpropagation
    # Calculate derivatives for output layer
    delta_a2 = error * derivative_sigmoid(a2)
    # Calculate derivatives for hidden layer
    delta_a1 = delta_a2 * w2.T * derivative_sigmoid(a1)

    # Update weights and biases
    w2_update = learning_rate * np.dot(a1.T, delta_a2)
    w1_update = learning_rate * np.dot(x.reshape(1, -1).T, delta_a1)
    b2_update = learning_rate * delta_a2
    b1_update = learning_rate * delta_a1

    return w1 + w1_update, w2 + w2_update, b1 + b1_update, b2 + b2_update

# Initialize weights and biases with random values
w1 = np.random.rand(2, 3)
w2 = np.random.rand(3, 1)
b1 = np.random.rand(1, 3)
b2 = np.random.rand()

# Learning parameters
learning_rate = 0.05
error_threshold = 0.002
max_iterations = 1000

# Training data
data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])
targets = np.array([0, 0, 0, 1])

# Training loop
for iteration in range(max_iterations):
    # Forward pass and calculate error
    error_sum = 0
    for i in range(len(data)):
        x = data[i]
        y = targets[i]
        w1, w2, b1, b2 = backpropagation(x, y, w1, w2, b1, b2, learning_rate)
        a2 = sigmoid(np.dot(sigmoid(np.dot(x, w1) + b1), w2) + b2)
        error_sum += np.abs(y - a2)

    # Check convergence
    if np.any(error_sum <= error_threshold):
        print("Converged after", iteration + 1, "iterations.")
        break

# Print final weights and biases
print("Final weights and biases:")
print("w1:", w1)
print("w2:", w2)
print("b1:", b1)
print("b2:", b2)

# Test the network with new inputs
new_input1 = np.array([0, 0])
new_input2 = np.array([1, 1])
output1 = sigmoid(np.dot(sigmoid(np.dot(new_input1, w1) + b1), w2) + b2)
output2 = sigmoid(np.dot(sigmoid(np.dot(new_input2, w1) + b1), w2) + b2)

print("Output for", new_input1, ":", output1)
print("Output for", new_input2, ":", output2)
