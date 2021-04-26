import numpy as np


def interchange(mat, row1, row2):
    """
        Row switching. A row within the matrix 'mat' will be switched with another row.
        
        Args:
            mat (numpy.matrix): the matrix to apply row switching transformation.
            row1 (int): the row index to be switched. The row index is in [0, mat.shape[0]-1]
            row2 (int): the another row index to be switched. The row index is in [0, mat.shape[0]-1]
    """


def scale(mat, row, scale):
    """
        Row multiplication. Each element in a given row will be multiplied by a non-zero constant. It is also known as scaling a row.

        Args:
            mat (numpy.matrix): the matrix to apply row multiplication transformation.
            row (int): the row index to be multiplied. The row index is in [0, mat.shape[0]-1]
            scale (int or float): the multiplier value.
    """


def add(mat, row1, row2, scale):
    """
        Row addition. A given row will be replaced by the sum of that row and a multiple of another row.
        Which mean (row2 of mat) = (row2 of mat) + scale * (row1 of mat).

        Args:
            mat (numpy.matrix): the matrix to apply row addition transformation.
            row1 (int): the row index1. The row index is in [0, mat.shape[0]-1]
            row2 (int): the row index2. The row index is in [0, mat.shape[0]-1]
            scale (int or float): the multiplier value.
    """


def row_simplest_trans(mat):
    """
        Transform mat to row simplest by elementary transformation.

        Args:
            mat (numpy.matrix): the matrix to apply elementary transformation.
    """


def gaussian_elimination(A, b):
    """
        Apply Gaussian Elimination to equation Ax=b.
        A is a non-singular square matrix. And b is a vector.

        Args:
            A (numpy.matrix): a non-singular square matrix with shape(m, m).
            b (numpy.ndarray): a vector with shape(m).

        Returns:
            numpy.ndarray: the solution vector x, with shape(m).
    """


def test1():
    globals_dict = globals()
    assert "interchange" in globals_dict, "function 'interchange' is not defined."
    assert "scale" in globals_dict, "function 'scale' is not defined."
    assert "add" in globals_dict, "function 'add' is not defined."
    import numpy as np

    test_mat = np.array([[1, 2, 3], [4, 5, 6], [0, 8, 9]])
    orig_mat = test_mat.copy()
    dest_mat1 = np.array([[1, 2, 3], [0, 8, 9], [4, 5, 6]])
    dest_mat2 = np.array([[4, 5, 6], [0, 8, 9], [1, 2, 3]])
    interchange(test_mat, 2, 1)
    assert np.all(
        test_mat == dest_mat1
    ), "wrong answer for calling 'interchange(mat, 2, 1)' s.t. mat =\n{}\nexpect:\n{}\nbut got:\n{}".format(
        orig_mat, dest_mat1, test_mat
    )
    orig_mat = test_mat.copy()
    interchange(test_mat, 2, 0)
    assert np.all(
        test_mat == dest_mat2
    ), "wrong answer for calling 'interchange(mat, 2, 0)' s.t. mat =\n{}\nexpect:\n{}\nbut got:\n{}".format(
        orig_mat, dest_mat2, test_mat
    )
    orig_mat = test_mat.copy()
    interchange(test_mat, 2, 2)
    assert np.all(
        test_mat == orig_mat
    ), "wrong answer for calling 'interchange(mat, 2, 2)' s.t. mat =\n{}\nexpect:\n{}\nbut got:\n{}".format(
        orig_mat, orig_mat, test_mat
    )

    print("--------------------------------------\nfunction 'interchange' check passed.")

    test_mat = np.array([[1, 2, 3], [4, 5, 6], [0, 8, 9]], dtype=np.float_)
    orig_mat = test_mat.copy()
    dest_mat1 = np.array([[2, 4, 6], [4, 5, 6], [0, 8, 9]], dtype=np.float_)
    dest_mat2 = np.array([[2, 4, 6], [6.0, 7.5, 9.0], [0, 8, 9]], dtype=np.float_)
    scale(test_mat, 0, 2.0)
    assert np.all(
        test_mat == dest_mat1
    ), "wrong answer for calling 'scale(mat, 0, 2.0)' s.t. mat =\n{}\nexpect:\n{}\nbut got:\n{}".format(
        orig_mat, dest_mat1, test_mat
    )
    orig_mat = test_mat.copy()
    scale(test_mat, 1, 1.5)
    assert np.all(
        test_mat == dest_mat2
    ), "wrong answer for calling 'scale(mat, 1, 1.5)' s.t. mat =\n{}\nexpect:\n{}\nbut got:\n{}".format(
        orig_mat, dest_mat2, test_mat
    )
    print("--------------------------------------\nfunction 'scale' check passed.")

    test_mat = np.array([[1, 2, 3], [4, 5, 6], [0, 8, 9]], dtype=np.float_)
    orig_mat = test_mat.copy()
    dest_mat1 = np.array([[1, 2, 3], [4, 5, 6], [0.5, 9.0, 10.5]], dtype=np.float_)
    add(test_mat, 0, 2, 0.5)
    assert np.all(
        test_mat == dest_mat1
    ), "wrong answer for calling 'add(mat, 0, 2, 0.5)' s.t. mat =\n{}\nexpect:\n{}\nbut got:\n{}".format(
        orig_mat, dest_mat1, test_mat
    )

    orig_mat = test_mat.copy()
    add(test_mat, 0, 2, 0.0)
    assert np.all(
        test_mat == orig_mat
    ), "wrong answer for calling 'add(mat, 0, 2, 0.0)' s.t. mat =\n{}\nexpect:\n{}\nbut got:\n{}".format(
        orig_mat, orig_mat, test_mat
    )
    print("--------------------------------------\nfunction 'add' check passed.")


def test2():
    import random
    import numpy.matlib

    # set seed
    random.seed(1)
    np.random.seed(1)

    # test 10000 times
    for _ in range(10000):
        # sample test mat
        m, n = random.randint(1, 10), random.randint(1, 10)
        test_mat = np.matlib.rand(m, n)

        # make test mat sparse
        for i in range(m):
            for j in range(n):
                if random.random() < 0.8:
                    test_mat[i, j] = 0

        orig_mat = test_mat.copy()
        # golden label
        np_rank = np.linalg.matrix_rank(test_mat)
        # test
        row_simplest_trans(test_mat)
        # comput rank
        rank = 0
        for row in test_mat:
            if np.any(row != 0):
                rank += 1
        assert (
            rank == np_rank
        ), "wrong answer for calling 'row_simplest_trans(mat)' s.t. mat =\n{}\nexpect rank:\n{}\nbut got a simplest mat:\n{}\nwith rank {}".format(
            orig_mat, np_rank, test_mat, rank
        )
    print(
        "--------------------------------------\nfunction 'row_simplest_trans' check passed."
    )


def test3():
    import random
    import numpy.matlib

    # set seed
    random.seed(1)
    np.random.seed(1)

    # test 200 times
    for _ in range(200):
        m = random.randint(1, 10)
        A_mat = np.matlib.rand(m, m)
        orig_mat = A_mat.copy()
        b = np.random.rand(m)

        np_rank = np.linalg.matrix_rank(A_mat)
        if np_rank != m:
            continue
        x = gaussian_elimination(A_mat, b)

        assert x.shape == (
            m,
        ), "Make sure 'gaussian_elimination' returns a vector with shape(m), expect shape({},), but got shape{}".format(
            m, x.shape
        )

        b = b.reshape(m, 1)
        np_x = (np.linalg.inv(orig_mat) @ b).reshape(m)
        assert np.all(
            (x - np_x) < 1e-10
        ), "wrong answer for calling 'gaussian_elimination(A, b)' s.t. A =\n{}\nb=\n{}\nexpect x:\n{}\nbut got x:\n{}".format(
            orig_mat, b, np_x, x
        )
    print(
        "--------------------------------------\nfunction 'gaussian_elimination' check passed."
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "test1":
            test1()
        elif sys.argv[1] == "test2":
            test2()
        elif sys.argv[1] == "test3":
            test3()

