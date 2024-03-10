from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load dataset
dataset_folder = r"C:\Users\91911\OneDrive\Documents\SEM-IV\MACHINE LEARNING\My Dataset"
class_labels = ["acrostic", "ballad"]

# Function to load dataset
def load_data(folder_path, class_labels):
    data = []
    labels = []
    for class_label in class_labels:
        class_folder = os.path.join(folder_path, class_label)
        files = os.listdir(class_folder)
        for file in files:
            with open(os.path.join(class_folder, file), 'r', encoding='utf-8') as f:
                text = f.read()
                data.append(text)
                labels.append(class_label)
    return data, labels

data, labels = load_data(dataset_folder, class_labels)

# Convert labels to numerical values
label_dict = {class_labels[i]: i for i in range(len(class_labels))}
y = np.array([label_dict[label] for label in labels])

# Convert text data to numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=200)  # You can adjust max_features as needed
X = tfidf_vectorizer.fit_transform(data)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='sgd', learning_rate_init=0.001, max_iter=1000)

# Train the model
mlp.fit(X_train, y_train)

# Evaluate model
train_accuracy = mlp.score(X_train, y_train)
test_accuracy = mlp.score(X_test, y_test)
print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Predict output for test set
predictions = mlp.predict(X_test)

# Display predictions
for i in range(X_test.shape[0]):
    print(f"Input: {X_test[i]}, Predicted Output: {class_labels[predictions[i]]}, True Label: {class_labels[y_test[i]]}")