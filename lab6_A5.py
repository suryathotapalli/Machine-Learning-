import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Define the data
data = np.array([
    [1, 20, 6, 2, 386],
    [1, 16, 3, 6, 289],
    [1, 27, 6, 2, 393],
    [1, 19, 1, 2, 110],
    [1, 24, 4, 2, 280],
    [1, 22, 1, 5, 167],
    [1, 15, 4, 2, 271],
    [1, 18, 4, 2, 274],
    [1, 21, 1, 4, 148],
    [1, 16, 2, 4, 198],
])

# Define the target labels
labels = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Initialize weights and learning rate
weights = np.random.rand(scaled_data.shape[1])
learning_rate = 0.1

# Define the activation function (ReLU)
def relu(x):
    return np.maximum(0, x)

# Training loop
for epoch in range(100):
    # Calculate the weighted sum of inputs
    z = np.dot(scaled_data, weights)

    # Apply the activation function
    y_pred = relu(z)

    # Calculate the error
    error = labels - y_pred

    # Update weights
    weights += learning_rate * np.dot(scaled_data.T, error)

# Print the final weights
print("Final weights:", weights)

# Use the trained perceptron to classify new data
new_data = np.array([1, 25, 5, 3, 400])
scaled_new_data = scaler.transform(new_data.reshape(1, -1))  # Reshape for consistent format

# Apply vectorized comparison to classify
if np.any(relu(np.dot(scaled_new_data, weights)) > 0):
    print("New transaction is classified as high value.")
else:
    print("New transaction is classified as low value.")
