import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def bipolar_step_function(x):
    return 1 if x >= 0 else -1

def step_function(x):
    return 1 if x >= 0 else 0

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def relu_function(x):
    return max(0, x)

def run_experiment(activation_function_name, activation_function, initial_weights):
    weights = np.array(initial_weights)
    errors = []  # List to store errors for each epoch
    epochs = []  # List to store epochs

    # Training data for XOR gate
    training_data = [
        (np.array([0, 0]), 0),
        (np.array([0, 1]), 1),
        (np.array([1, 0]), 1),
        (np.array([1, 1]), 0)
    ]

    # Loop for a maximum of 1000 epochs
    for epoch in range(1000):
        total_error = 0
        for x, y in training_data:
            # Calculate the weighted sum
            weighted_sum = np.dot(x, weights[1:]) + weights[0]

            # Apply activation function
            output = activation_function(weighted_sum)

            # Calculate the error
            error = y - output
            total_error += error**2

            # Update weights based on learning rule
            weights[0] += learning_rate * error
            weights[1:] += learning_rate * error * x

        # Calculate average error for the epoch
        average_error = total_error / len(training_data)
        errors.append(average_error)
        epochs.append(epoch)

        # Stop training if convergence criteria is met
        if average_error <= 0.002:
            print(f"Converged after {epoch+1} epochs for {activation_function_name}")
            break

    # Plot the error vs epochs
    plt.plot(epochs, errors, label=activation_function_name)

# Initialize weights and learning rate
learning_rate = 0.05

# Run experiments with different activation functions
initial_weights = [10, 0.2, -0.75]
run_experiment("Bipolar Step", bipolar_step_function, initial_weights.copy())

initial_weights = [10, 0.2, -0.75]
run_experiment("Step", step_function, initial_weights.copy())

initial_weights = [10, 0.2, -0.75]
run_experiment("Sigmoid", sigmoid_function, initial_weights.copy())

initial_weights = [10, 0.2, -0.75]
run_experiment("ReLU", relu_function, initial_weights.copy())

# Show the plot
plt.xlabel("Epochs")
plt.ylabel("Sum-Squared Error")
plt.title("Sum-Squared Error vs Epochs (XOR Gate)")
plt.grid(True)
plt.legend()
plt.show()
