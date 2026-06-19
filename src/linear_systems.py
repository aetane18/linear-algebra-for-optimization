import numpy as np


def augmented_matrix(A, b):
    """
    Create the augmented matrix [A | b].

    Example:
    A = [[1, 2],
         [3, 4]]

    b = [5, 6]

    [A | b] = [[1, 2, 5],
               [3, 4, 6]]
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)

    return np.hstack((A, b))


def classify_system(A, b):
    """
    Classify a linear system Ax = b.

    Returns one of:
    - "unique solution"
    - "no solution"
    - "infinitely many solutions"
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    rows, cols = A.shape

    Ab = augmented_matrix(A, b)

    rank_A = np.linalg.matrix_rank(A)
    rank_Ab = np.linalg.matrix_rank(Ab)

    if rank_A == rank_Ab == cols:
        return "unique solution"

    elif rank_A == rank_Ab and rank_A < cols:
        return "infinitely many solutions"

    else:
        return "no solution"


def solve_if_unique(A, b):
    """
    Solve Ax = b only if the system has a unique solution.
    """
    system_type = classify_system(A, b)

    if system_type != "unique solution":
        raise ValueError(f"Cannot solve directly because system has {system_type}")

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    return np.linalg.solve(A, b)


def condition_number(A):
    """
    Return the condition number of matrix A.

    Large condition number means the system is sensitive to small changes.
    """
    A = np.array(A, dtype=float)
    return np.linalg.cond(A)


def is_ill_conditioned(A, threshold=1000):
    """
    Check whether matrix A is ill-conditioned.

    A high condition number means small input changes can cause large output changes.
    """
    cond = condition_number(A)
    return cond > threshold


if __name__ == "__main__":
    # -----------------------------
    # 1. Unique solution example
    # -----------------------------
    A_unique = [
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ]

    b_unique = [8, -11, -3]

    print("Unique Solution Example")
    print("System type:", classify_system(A_unique, b_unique))
    print("Solution:", solve_if_unique(A_unique, b_unique))
    print()

    # -----------------------------
    # 2. No solution example
    # -----------------------------
    A_no_solution = [
        [1, 1],
        [2, 2]
    ]

    b_no_solution = [3, 8]

    print("No Solution Example")
    print("System type:", classify_system(A_no_solution, b_no_solution))
    print()

    # -----------------------------
    # 3. Infinite solutions example
    # -----------------------------
    A_infinite = [
        [1, 1],
        [2, 2]
    ]

    b_infinite = [3, 6]

    print("Infinite Solutions Example")
    print("System type:", classify_system(A_infinite, b_infinite))
    print()

    # -----------------------------
    # 4. Ill-conditioned example
    # -----------------------------
    A_ill = [
        [1, 1],
        [1, 1.0001]
    ]

    b_ill = [2, 2.0001]

    print("Ill-Conditioned Example")
    print("System type:", classify_system(A_ill, b_ill))
    print("Condition number:", condition_number(A_ill))
    print("Is ill-conditioned?", is_ill_conditioned(A_ill))
    print("Solution:", solve_if_unique(A_ill, b_ill))