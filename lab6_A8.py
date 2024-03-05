import numpy as np
import matplotlib.pyplot as plt

# Initialize weights and learning rate
w0 = 10
w1 = 0.2
w2 = -0.75
learning_rate = 0.05

# Define the user-defined activation function (replace with your formula)
def custom_activation_function(x):
    # Replace this with the mathematical formula of the activation function
    # For example, using a step function:
    return 1 if x >= 0 else 0

# Training data for XOR gate
training_data = [
    (np.array([0, 0]), 0),
    (np.array([0, 1]), 1),
    (np.array([1, 0]), 1),
    (np.array([1, 1]), 0)
]

errors = []  # List to store errors for each epoch
epochs = []  # List to store epochs

# Loop for a maximum of 1000 epochs
for epoch in range(1000):
    total_error = 0
    for x, y in training_data:
        # Calculate the weighted sum
        weighted_sum = w0 + (w1 * x[0]) + (w2 * x[1])
        # Apply custom activation function
        output = custom_activation_function(weighted_sum)
        # Calculate the error
        error = y - output
        total_error += error**2

        # Update weights based on the learning rule
        w0 += learning_rate * error
        w1 += learning_rate * error * x[0]
        w2 += learning_rate * error * x[1]

    # Calculate average error for the epoch
    average_error = total_error / len(training_data)
    errors.append(average_error)
    epochs.append(epoch)

    # Stop training if convergence criteria are met
    if average_error <= 0.002:
        print(f"Converged after {epoch+1} epochs")
        break

# Plot the error vs epochs
plt.plot(epochs, errors)
plt.xlabel("Epochs")
plt.ylabel("Sum-Squared Error")
plt.title("Sum-Squared Error vs Epochs")
plt.grid(True)
plt.show()

# Test the perceptron with unseen data (modify for your dataset)
test_data = [
    np.array([0, 0]),
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1])
]

for x in test_data:
    # Calculate the weighted sum
    weighted_sum = w0 + (w1 * x[0]) + (w2 * x[1])
    # Apply custom activation function
    output = custom_activation_function(weighted_sum)
    print(f"Input: {x}, Output: {output}")
