# written by Souritra Garai
# Date - 2nd July 2020
import numpy as np

def first_non_zero_element(M, n) :

    # finding the indices of non-zero elements
    # and dividing them into row index and column index
    non_zero_rows, non_zero_cols = np.nonzero(M[:, 0:n])

    # print('rows\n', non_zero_rows)
    # print('cols\n', non_zero_cols)

    # finding the first appearance of each row index
    non_zero_rows -= np.append(0, non_zero_rows[0: -1])
    
    non_zero_rows[0] = 1

    # print('first rows\n', non_zero_rows)

    col_indices = np.nonzero(non_zero_rows)[0]
    # print('col indices\n', col_indices)

    # returning the column index corresponding to
    # the first appearance of each row index
    return non_zero_cols[col_indices]

def LUDecomposition(A, n) :

    L = np.zeros((n, n))
    U = np.copy(A)

    # Iterating for each row # i
    for i in range(n-1) :

        # for each lower row j
        # we need to multiply the modified top row 
        # with the left-most element of j
        # outer(left-most column, top row)
        # = matrix that needs to be subtracted from the matrix of lower rows
        # M[i+1:, :] -= np.multiply.outer(M[i+1:, i], M[i, :])

        L[i+1:, i] = U[i+1:, i] / U[i, i]

        U[i+1:, :] -= np.multiply.outer(L[i+1:, i], U[i, :])

        # print(i, '\n', L, '\n', U)

    np.fill_diagonal(L, 1)

    return L, U

def solveLdb(L, b, n) :

    d = np.zeros(n)

    d[0] = b[0]

    for i in range(1, n) :

        d[i] = b[i] - np.inner(L[i, 0:i], d[0:i])

    return d

def solveUxd(U, d, n) :

    x = np.zeros(n)

    x[n-1] = d[n-1] / U[n-1, n-1]
    
    for i in range(n-2, -1, -1) :

        x[i] = ( d[i] - np.inner(U[i, i+1:], x[i+1:]) ) / U[i, i]

    return x

def vectorised_SolveLUDecomposition(A, B, n) :

    # We are preparing the matrix by swapping rows
    # so that row with left most non-zero value appears at top

    # finding the index of left most non zero element
    # for each row
    first_non_zero_element_index = np.array(first_non_zero_element(A, n))

    sorted_indices = first_non_zero_element_index.argsort()

    A = A[sorted_indices]
    B = B[:, sorted_indices]

    L, U = LUDecomposition(A, n)

    # print(L)
    # print(U)

    D = np.array([ solveLdb(L, b, n) for b in B ])
    # print(D)
    X = np.array([ solveUxd(U, d, n) for d in D ])

    return X

def SolveLUDecomposition(A, b, n) :

    # We are preparing the matrix by swapping rows
    # so that row with left most non-zero value appears at top

    # finding the index of left most non zero element
    # for each row
    first_non_zero_element_index = np.array(first_non_zero_element(A, n))

    sorted_indices = first_non_zero_element_index.argsort()

    A = A[sorted_indices]
    b = b[sorted_indices]

    L, U = LUDecomposition(A, n)

    # print(L)
    # print(U)

    d = solveLdb(L, b, n)
    # print(D)
    x = solveUxd(U, d, n)

    return x

if __name__ == "__main__":
    
    A = np.array([  [   5,  6,  10   ],
                    [   7,  11,  9   ],
                    [   19,  3,  2   ]   ], dtype=float)
    
    b = np.array([1, 2, 3], dtype=float)

    print(np.linalg.solve(A,b))

    print(SolveLUDecomposition(A,b,3))