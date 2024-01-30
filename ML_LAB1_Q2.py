def get_matrix():
   
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))
    
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            element = float(input(f"Enter the element at position ({i+1}, {j+1}): "))
            row.append(element)
        matrix.append(row)
    
    return matrix

def multiply_matrices(matrix_A, matrix_B):
    
    rows_A, cols_A = len(matrix_A), len(matrix_A[0])
    rows_B, cols_B = len(matrix_B), len(matrix_B[0])
    
    if cols_A != rows_B:
        print("Error: Matrices are not multipliable.")
        return None
    
    result_matrix = []
    for i in range(rows_A):
        row = []
        for j in range(cols_B):
            element = sum(matrix_A[i][k] * matrix_B[k][j] for k in range(cols_A))
            row.append(element)
        result_matrix.append(row)
    
    return result_matrix

def main():
    print("Enter Matrix A:")
    matrix_A = get_matrix()
    
    print("\nEnter Matrix B:")
    matrix_B = get_matrix()
    
    result = multiply_matrices(matrix_A, matrix_B)
    
    if result is not None:
        print("\nMatrix A * Matrix B:")
        for row in result:
            print(row)

if __name__ == "__main__":
    main()
