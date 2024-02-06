def euclidean_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimension")

    squared_sum = sum((v1 - v2) ** 2 for v1, v2 in zip(vector1, vector2))
    return squared_sum ** 0.5  # square root

def manhattan_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimension")

    return sum(abs(v1 - v2) for v1, v2 in zip(vector1, vector2))

# Example usage:
vector_a = [1, 2, 3]
vector_b = [4, 5, 6]

euclidean_result = euclidean_distance(vector_a, vector_b)
manhattan_result = manhattan_distance(vector_a, vector_b)

print("Euclidean Distance:", euclidean_result)
print("Manhattan Distance:", manhattan_result)
