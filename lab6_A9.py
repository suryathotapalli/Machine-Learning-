import numpy as np
import matplotlib.pyplot as plt

# Define sigmoid activation function
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

# Initialize weights and learning rate
w0, w1, w2, w3, w4, w5 = 0.5, 0.5, 0.5, 0.5, 0.5, 0.5  # or any initial values you prefer
learning_rate = 0.05

# Training data for OR gate (modify for your dataset)
training_data = [
    (np.array([0, 0]), np.array([1, 0])),  # OR (0, 0) = (1, 0)
    (np.array([0, 1]), np.array([1, 0])),  # OR (0, 1) = (1, 0)
    (np.array([1, 0]), np.array([1, 0])),  # OR (1, 0) = (1, 0)
    (np.array([1, 1]), np.array([0, 1])),  # OR (1, 1) = (0, 1)
]

errors = []  # List to store errors for each epoch
epochs = []  # List to store epochs

# Loop for a maximum of 1000 epochs
for epoch in range(1000):
    total_error = 0
    for x, y in training_data:
        # Calculate the weighted sum for each output node
        weighted_sum1 = np.dot(np.array([w0, w1, w2]), np.concatenate(([1], x)))  # Add bias term
        weighted_sum2 = np.dot(np.array([w3, w4, w5]), np.concatenate(([1], x)))  # Add bias term

        # Apply sigmoid activation function to each output node
        output1 = sigmoid_function(weighted_sum1)
        output2 = sigmoid_function(weighted_sum2)

        # Calculate the error for each output node
        error1 = y[0] - output1
        error2 = y[1] - output2
        total_error += error1**2 + error2**2

        # Update weights based on learning rule
        w0 += learning_rate * error1 * 1  # Bias weight update
        w1 += learning_rate * error1 * x[0]
        w2 += learning_rate * error1 * x[1]
        w3 += learning_rate * error2 * 1  # Bias weight update
        w4 += learning_rate * error2 * x[0]
        w5 += learning_rate * error2 * x[1]

    # Calculate average error for the epoch
    average_error = total_error / len(training_data)
    errors.append(average_error)
    epochs.append(epoch)

    # Stop training if convergence criteria is met
    if average_error <= 0.002:
        print(f"Converged after {epoch+1} epochs")
        break

# Plot the error vs epochs
plt.plot(epochs, errors)
plt.xlabel("Epochs")
plt.ylabel("Sum-Squared Error")
plt.title("Sum-Squared Error vs Epochs (OR Gate)")
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
    # Calculate weighted sums for each output node
    weighted_sum1 = np.dot(np.array([w0, w1, w2]), np.concatenate(([1], x)))  # Add bias term
    weighted_sum2 = np.dot(np.array([w3, w4, w5]), np.concatenate(([1], x)))  # Add bias term

    # Apply sigmoid activation function
    output1 = sigmoid_function(weighted_sum1)
    output2 = sigmoid_function(weighted_sum2)

    # Interpret the output vector based on thresholds (replace)
    print(f"Input: {x}, Output: ({output1}, {output2})")
