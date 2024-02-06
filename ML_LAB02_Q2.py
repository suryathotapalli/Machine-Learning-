import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def k_nearest_neighbors(X_train, y_train, x_test, k=3):
    distances = [euclidean_distance(x_test, x) for x in X_train]
    k_neighbors_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_neighbors_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

# Example usage:
X_train = np.array([[1, 2], [2, 3], [3, 1], [4, 2]])
y_train = np.array([0, 0, 1, 1])

x_test = np.array([2.5, 2])
k_value = 2

predicted_label = k_nearest_neighbors(X_train, y_train, x_test, k=k_value)
print(f"The predicted label for the test instance is: {predicted_label}")
