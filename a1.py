import numpy as np
import pandas as pd
df = pd.read_excel('Lab Session1 Data.xlsx', sheet_name='Purchase Data')
A = df.iloc[:, 1:4]
C = df.iloc[:, 4]
dimensionality = A.shape[1]
num_vectors = A.shape[0]
rank_A = np.linalg.matrix_rank(A)
A_pseudo_inverse = np.linalg.pinv(A)
cost_per_product = np.dot(A_pseudo_inverse, C)
print("Dimensionality of the vector space:", dimensionality)
print("Number of vectors in the vector space:", num_vectors)
print("Rank of Matrix A:", rank_A)
print("Cost of each product using Pseudo-Inverse:")
print(cost_per_product)