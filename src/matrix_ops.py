def shape(A):
    """
    Return the shape of matrix A as (rows, columns).
    """
    rows = len(A)
    cols = len(A[0])
    return rows, cols


def matrix_add(A, B):
    """
    Add two matrices A and B.
    A and B must have the same shape.
    """
    rows_A, cols_A = shape(A)
    rows_B, cols_B = shape(B)

    if rows_A != rows_B or cols_A != cols_B:
        raise ValueError("Matrices must have the same shape")

    result = []

    for i in range(rows_A):
        row = []

        for j in range(cols_A):
            value = A[i][j] + B[i][j]
            row.append(value)

        result.append(row)

    return result


def matrix_subtract(A, B):
    """
    Subtract matrix B from matrix A.
    A and B must have the same shape.
    """
    rows_A, cols_A = shape(A)
    rows_B, cols_B = shape(B)

    if rows_A != rows_B or cols_A != cols_B:
        raise ValueError("Matrices must have the same shape")

    result = []

    for i in range(rows_A):
        row = []

        for j in range(cols_A):
            value = A[i][j] - B[i][j]
            row.append(value)

        result.append(row)

    return result


def scalar_multiply(c, A):
    """
    Multiply every entry of matrix A by scalar c.
    """
    rows, cols = shape(A)

    result = []

    for i in range(rows):
        row = []

        for j in range(cols):
            value = c * A[i][j]
            row.append(value)

        result.append(row)

    return result


def transpose(A):
    """
    Return the transpose of matrix A.
    Rows become columns.
    Columns become rows.
    """
    rows, cols = shape(A)

    result = []

    for j in range(cols):
        row = []

        for i in range(rows):
            value = A[i][j]
            row.append(value)

        result.append(row)

    return result


if __name__ == "__main__":
    A = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    B = [
        [9, 8, 7],
        [6, 5, 4],
        [3, 2, 1]
    ]

    print("Shape of A:")
    print(shape(A))

    print("\nA + B:")
    print(matrix_add(A, B))

    print("\nA - B:")
    print(matrix_subtract(A, B))

    print("\n3A:")
    print(scalar_multiply(3, A))

    print("\nTranspose of A:")
    print(transpose(A))