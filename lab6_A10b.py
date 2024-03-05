from sklearn.neural_network import MLPClassifier
import numpy as np

# Define XOR gate inputs and corresponding outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Initialize MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=1000)

# Train the model
mlp.fit(X, y)

# Predict output for XOR gate inputs
predictions = mlp.predict(X)

# Display predictions
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted Output: {predictions[i]}")
