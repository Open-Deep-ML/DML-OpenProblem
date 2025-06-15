import numpy as np


def translate_object(points, tx, ty):
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    homogeneous_points = np.hstack([np.array(points), np.ones((len(points), 1))])

    translated_points = np.dot(homogeneous_points, translation_matrix.T)

    return translated_points[:, :2].tolist()
