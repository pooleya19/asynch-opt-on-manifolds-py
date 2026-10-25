import numpy as np


def copy_manifold_point(manifold_point):
    if type(manifold_point) == np.ndarray:
        manifold_point_copy = manifold_point.copy()
    elif type(manifold_point) == list:
        manifold_point_copy = [copy_manifold_point(manifold_point[i]) for i in range(len(manifold_point))]
    elif manifold_point is None:
        manifold_point_copy = None
    else:
        raise ValueError("Attempted to copy unknown type (%s)." % str(type(manifold_point)))
    return manifold_point_copy


def scale_tangent_vector(tangent_vector, scalar):
    if type(tangent_vector) == np.ndarray:
        tangent_vector_scaled = tangent_vector * scalar
    elif type(tangent_vector) == list:
        tangent_vector_scaled = [scale_tangent_vector(tangent_vector[i], scalar) for i in range(len(tangent_vector))]
    elif tangent_vector is None:
        tangent_vector_scaled = None
    else:
        raise ValueError("Attempted to scale unknown type (%s)." % str(type(tangent_vector)))
    return tangent_vector_scaled


def sum_tangent_vectors(tangent_vector1, tangent_vector2):
    if type(tangent_vector1) != type(tangent_vector2):
        raise ValueError("Attempted to add unequal types (%s, %s)." % (str(type(tangent_vector1)), str(type(tangent_vector2))))
    if type(tangent_vector1) == np.ndarray:
        tangent_vector_sum = tangent_vector1 + tangent_vector2
    elif type(tangent_vector1) == list:
        if len(tangent_vector1) != len(tangent_vector2):
            raise ValueError("Attempted to add lists of unequal lengths (%g, %g)." % (len(tangent_vector1), len(tangent_vector2)))
        tangent_vector_sum = [sum_tangent_vectors(tangent_vector1[i], tangent_vector2[i]) for i in range(len(tangent_vector1))]
    elif tangent_vector1 is None:
        tangent_vector_sum = None
    else:
        raise ValueError("Attempted to scale unknown type (%s)." % str(type(tangent_vector1)))
    return tangent_vector_sum


def freeze_manifold_point(manifold_point):
    if type(manifold_point) == np.ndarray:
        manifold_point.flags.writeable = False
    elif type(manifold_point) == list:
        for i in range(len(manifold_point)):
            freeze_manifold_point(manifold_point[i])
    elif manifold_point is None:
        pass
    else:
        raise ValueError("Attempted to freeze unknown type (%s)." % str(type(manifold_point)))
    return manifold_point