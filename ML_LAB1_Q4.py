def get_matrix_input():
    
    num_rows = int(input("Enter the number of rows in the matrix: "))
    num_cols = int(input("Enter the number of columns in the matrix: "))

    matrix = []

    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            element = float(input(f"Enter element at position ({i+1}, {j+1}): "))
            row.append(element)
        matrix.append(row)

    return matrix

def transpose_matrix(matrix):

    transposed_matrix = [list(row) for row in zip(*matrix)]
    return transposed_matrix

def main():

    input_matrix = get_matrix_input()

    transposed_result = transpose_matrix(input_matrix)

    print("\nOriginal Matrix:")
    for row in input_matrix:
        print(row)

    print("\nTransposed Matrix:")
    for row in transposed_result:
        print(row)

if __name__ == "_main_":
    main()