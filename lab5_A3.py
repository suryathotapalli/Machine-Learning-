import numpy as np
import matplotlib.pyplot as plt

# Generate 20 data points with 2 features (X & Y) randomly between 1 and 10
np.random.seed(0)  # for reproducibility
X = np.random.randint(1, 11, size=(20, 2))

# Let's say class0 is Blue and class1 is Red
classes = np.random.randint(0, 2, size=20)

# Separate points for each class
class0_X = X[classes == 0]
class1_X = X[classes == 1]

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(class0_X[:, 0], class0_X[:, 1], color='blue', label='Class 0')
plt.scatter(class1_X[:, 0], class1_X[:, 1], color='red', label='Class 1')
plt.title('Scatter Plot of Training Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
