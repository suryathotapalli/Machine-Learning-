import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class MyDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def information_gain(self, X, y, feature_index, threshold):
        # Split data based on the given feature and threshold
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        # Calculate entropy before split
        entropy_before = self.entropy(y)

        # Calculate entropy after split
        entropy_left = self.entropy(y[left_mask])
        entropy_right = self.entropy(y[right_mask])
        entropy_after = (len(y[left_mask]) / len(y)) * entropy_left + (len(y[right_mask]) / len(y)) * entropy_right

        # Calculate information gain
        information_gain = entropy_before - entropy_after

        return information_gain

    def find_best_split(self, X, y):
        best_information_gain = -1
        best_feature_index = None
        best_threshold = None

        for feature_index in range(X.shape[1]):
            unique_values = np.unique(X[:, feature_index])
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                information_gain = self.information_gain(X, y, feature_index, threshold)
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def build_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            leaf_node = {
                'type': 'leaf',
                'class': np.argmax(np.bincount(y)),
                'samples': len(y),
                'entropy': self.entropy(y)
            }
            return leaf_node

        feature_index, threshold = self.find_best_split(X, y)

        if feature_index is None:
            leaf_node = {
                'type': 'leaf',
                'class': np.argmax(np.bincount(y)),
                'samples': len(y),
                'entropy': self.entropy(y)
            }
            return leaf_node

        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        decision_node = {
            'type': 'decision',
            'feature_index': feature_index,
            'threshold': threshold,
            'left_subtree': left_subtree,
            'right_subtree': right_subtree
        }

        return decision_node

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_instance(self, instance, tree):
        if tree['type'] == 'leaf':
            return tree['class']
        else:
            if instance[tree['feature_index']] <= tree['threshold']:
                return self.predict_instance(instance, tree['left_subtree'])
            else:
                return self.predict_instance(instance, tree['right_subtree'])

    def predict(self, X):
        predictions = [self.predict_instance(instance, self.tree) for instance in X]
        return np.array(predictions)

    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)


# Example usage:
poem_csv = r"G:\My Drive\Google Drive Documents\Sneha Saragadam\Sneha Engineering Plan 2022-2025\Engineering Preparation\2nd Year\machinelearning\project\poems_data.csv"
df = pd.read_csv(poem_csv)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train the custom Decision Tree model
my_decision_tree = MyDecisionTree(max_depth=5)
my_decision_tree.fit(X_train, y_train)

# Make predictions
y_pred_train = my_decision_tree.predict(X_train)
y_pred_test = my_decision_tree.predict(X_test)

# Calculate accuracy
train_accuracy = my_decision_tree.accuracy(y_train, y_pred_train)
test_accuracy = my_decision_tree.accuracy(y_test, y_pred_test)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)

# Output performance metrics
print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)
