import pandas as pd

# Assuming you have a DataFrame 'df' with a column named 'category_column'
data = {'category_column': ['A', 'B', 'A', 'C', 'B', 'C']}
df = pd.DataFrame(data)

def one_hot_encode_categorical(data, column_name):
    """
    Convert categorical variable to numeric using one-hot encoding.

    Parameters:
    - data: Pandas DataFrame, the dataset containing the categorical variable.
    - column_name: str, the name of the column to be one-hot encoded.

    Returns:
    - data: Pandas DataFrame, the dataset with the one-hot encoded columns.
    - one_hot_mapping: dict, a mapping of original labels to corresponding one-hot encoded columns.
    """

    # Extract unique labels from the column
    unique_labels = data[column_name].unique()

    # Create one-hot encoding mapping
    one_hot_mapping = {label: [0] * len(unique_labels) for label in unique_labels}
    for i, label in enumerate(unique_labels):
        one_hot_mapping[label][i] = 1

    # Create new columns for one-hot encoding
    for i, label in enumerate(unique_labels):
        data[f'{column_name}_{i}'] = data[column_name].apply(lambda x: one_hot_mapping[x][i])

    # Drop the original categorical column
    data = data.drop(column_name, axis=1)

    return data, one_hot_mapping

# Example usage:
df, one_hot_mapping = one_hot_encode_categorical(df, 'category_column')

# Access the one-hot encoding mapping if needed
print("One-Hot Encoding Mapping:")
print(one_hot_mapping)
print("DataFrame after One-Hot Encoding:")
print(df)
 