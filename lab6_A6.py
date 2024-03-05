import numpy as np

# Define the data (same as perceptron example)
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

# Define the target labels (same as perceptron example)
labels = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])

# Calculate the pseudoinverse (Moore-Penrose) using np.linalg.pinv
weights_pseudoinverse = np.linalg.pinv(data) @ labels

# Use the weights to predict for the new data (same as perceptron example)
new_data = np.array([1, 25, 5, 3, 400])
predicted_label = np.dot(new_data, weights_pseudoinverse)

if predicted_label > 0.5:
  print("New transaction (using pseudoinverse) is classified as high value.")
else:
  print("New transaction (using pseudoinverse) is classified as low value.")
