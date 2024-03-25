import numpy as np
import numexpr as ne

def distance_matrix_numpy(points):
    r_i = points[:, np.newaxis]
    r_j = points[np.newaxis, :]
    d_ij = np.sqrt(((r_j - r_i) ** 2).sum(axis=2))
    return d_ij

def distance_matrix_numexpr(points):
    r_i = points[:, np.newaxis]
    r_j = points[np.newaxis, :]
    # numexpr does not directly support broadcasting in the same way as NumPy,
    # so we calculate the squared differences and then sum and sqrt separately.
    d_ij_squared = ne.evaluate("sum((r_j - r_i) ** 2, axis=2)")
    d_ij = ne.evaluate("sqrt(d_ij_squared)")
    return d_ij

points = np.random.rand(1000, 2)