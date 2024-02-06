import pandas as pd

# Assuming you have a DataFrame 'df' with a column named 'category_column'
data = {'category_column': ['A', 'B', 'A', 'C', 'B', 'C']}
df = pd.DataFrame(data)

def label_encode_categorical(data, column_name):
    """
    Convert categorical variable to numeric using label encoding.

    Parameters:
    - data: Pandas DataFrame, the dataset containing the categorical variable.
    - column_name: str, the name of the column to be label encoded.

    Returns:
    - data: Pandas DataFrame, the dataset with the label encoded column.
    - label_mapping: dict, a mapping of original labels to corresponding numeric values.
    """

    # Extract unique labels from the column
    unique_labels = data[column_name].unique()

    # Create a mapping of labels to numeric values
    label_mapping = {label: i for i, label in enumerate(unique_labels)}

    # Apply label encoding to the column
    data[column_name] = data[column_name].map(label_mapping)

    return data, label_mapping

# Example usage:
df, label_mapping = label_encode_categorical(df, 'category_column')

# Access the label mapping if needed
print("Label Mapping:", label_mapping)
print("DataFrame after Label Encoding:")
print(df)
