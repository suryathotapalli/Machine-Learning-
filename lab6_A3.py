import numpy as np
import matplotlib.pyplot as plt

# Initialize weights and learning rates
w0 = 10
w1 = 0.2
w2 = -0.75
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Function to train the perceptron for a given learning rate
def train_perceptron(learning_rate):
    # Copy the initial weights
    current_w0, current_w1, current_w2 = w0, w1, w2

    # Training data for AND gate (modify for your dataset)
    training_data = [
        (np.array([0, 0]), 0),
        (np.array([0, 1]), 0),
        (np.array([1, 0]), 0),
        (np.array([1, 1]), 1)
    ]

    errors = []  # List to store errors for each epoch
    epochs = 0  # Variable to store the number of iterations

    # Loop for a maximum of 1000 epochs
    while epochs < 1000:
        total_error = 0
        for x, y in training_data:
            # Calculate the weighted sum
            weighted_sum = current_w0 + (current_w1 * x[0]) + (current_w2 * x[1])
            # Apply custom activation function
            output = 1 if weighted_sum >= 0 else 0
            # Calculate the error
            error = y - output
            total_error += error ** 2

            # Update weights based on the learning rule
            current_w0 += learning_rate * error
            current_w1 += learning_rate * error * x[0]
            current_w2 += learning_rate * error * x[1]

        # Calculate average error for the epoch
        average_error = total_error / len(training_data)
        errors.append(average_error)

        # Stop training if convergence criteria are met
        if average_error <= 0.002:
            print(f"Converged after {epochs+1} epochs with learning rate {learning_rate}")
            break

        epochs += 1

    return epochs

# Train the perceptron for each learning rate
convergence_iterations = []
for lr in learning_rates:
    iterations = train_perceptron(lr)
    convergence_iterations.append(iterations)

# Plot the number of iterations vs learning rates
plt.plot(learning_rates, convergence_iterations, marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("Number of Iterations to Converge")
plt.title("Number of Iterations vs Learning Rate")
plt.grid(True)
plt.show()
