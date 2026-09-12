import numpy as np


def copy_manifold_point(manifold_point):
    if type(manifold_point) == np.ndarray:
        manifold_point_copy = manifold_point.copy()
    elif type(manifold_point) == list:
        manifold_point_copy = []
        for i in range(len(manifold_point)):
            manifold_point_copy.append(copy_manifold_point(manifold_point[i]))
    elif manifold_point is None:
        manifold_point_copy = None
    else:
        raise ValueError("Attempted to copy unknown type (%s)." % str(type(manifold_point)))
    return manifold_point_copy


def scale_manifold_point(manifold_point, scalar):
    if type(manifold_point) == np.ndarray:
        manifold_point_scaled = manifold_point * scalar
    elif type(manifold_point) == list:
        manifold_point_scaled = []
        for i in range(len(manifold_point)):
            manifold_point_scaled.append(copy_manifold_point(manifold_point[i]))
    else:
        raise ValueError("Attempted to copy unknown type (%s)." % str(type(manifold_point)))
    return manifold_point_scaled