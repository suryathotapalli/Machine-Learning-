import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def equal_width_binning(data, num_bins):
    # Perform equal width binning for continuous valued features
    bin_width = (data.max() - data.min()) / num_bins
    bins = [data.min() + i * bin_width for i in range(num_bins + 1)]
    return pd.cut(data, bins=bins, labels=False)

def equal_frequency_binning(data, num_bins):
    # Perform equal frequency binning for continuous valued features
    bins = pd.qcut(data, q=num_bins, labels=False, duplicates='drop')
    return bins

def detect_root_feature(poem_csv, binning_type='equal_width', num_bins=5):
    # Load the poem embeddings CSV file
    df = pd.read_csv(poem_csv)

    # Assume the last column is the target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Convert continuous features to categorical by binning
    if binning_type == 'equal_width':
        X_binned = X.apply(lambda x: equal_width_binning(x, num_bins), axis=0)
    elif binning_type == 'equal_frequency':
        X_binned = X.apply(lambda x: equal_frequency_binning(x, num_bins), axis=0)
    else:
        raise ValueError("Invalid binning_type. Choose 'equal_width' or 'equal_frequency'.")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_binned, y, test_size=0.2, random_state=42)

    # Initialize Decision Tree Classifier
    clf = DecisionTreeClassifier(criterion="entropy")

    # Fit the classifier
    clf.fit(X_train, y_train)

    # Get feature importances
    feature_importances = clf.feature_importances_

    # Find the index of the feature with the highest importance
    root_feature_index = feature_importances.argmax()

    # Get the name of the root feature
    root_feature = df.columns[root_feature_index]

    # Get the binned values of the root feature
    root_feature_binned = X_binned.iloc[:, root_feature_index]

    return root_feature, root_feature_binned

# Example usage:
poem_csv = r"G:\My Drive\Google Drive Documents\Sneha Saragadam\Sneha Engineering Plan 2022-2025\Engineering Preparation\2nd Year\machinelearning\project\poems_data.csv"
root_feature, root_feature_binned = detect_root_feature(poem_csv, binning_type='equal_width', num_bins=5)
print("Root feature for Decision Tree:", root_feature)
print("Binned values for root feature:")
print(root_feature_binned)
