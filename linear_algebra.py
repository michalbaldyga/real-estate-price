from typing import List

Vector = List[float]
Matrix = List[Vector]


def get_column(x: Matrix, col: int):
    """Returns a specific matrix column"""
    return [x_i[col] for x_i in x]


def create_extended_matrix(x: Matrix) -> Matrix:
    """Returns a matrix in form:
    [[1, x_11, x_21, x_31, ...]
     [1, x_12, x_22, x_32, ...]
     [1, x_13, x_23, x_33, ...]]
    """
    ext_mat = []
    for i in range(len(x[0])):
        new_row = [1]
        for j in range(len(x)):
            new_row.append(x[j][i])
        ext_mat.append(new_row)
    return ext_mat
