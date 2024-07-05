import numpy as np


def set_pia_zero(points, pia_grid):
    pia_y = pia_grid.mean(axis=0)[1]
    return np.hstack([points[:, [0]], points[:, [1]] - pia_y, points[:, [2]]])


def rotate_points(points, axis, degree=5):
    """Rotates points in the MICrONS dataset.
    Adapted from code provided by Sven Dorkenwald & Forrest Collman.

    Args:
        points: Nx3 numpy array
            coordinates in nm
        degree: int
            degrees of rotation

    Returns:
        corrected_points: Nx3 numpy array
            rotated points
    """
    angle = degree * np.pi / 180
    corrected_points = points.copy()

    if axis == 0:
        y = points[..., 1] * np.cos(angle) - points[..., 2] * np.sin(angle)
        z = points[..., 1] * np.sin(angle) + points[..., 2] * np.cos(angle)
        corrected_points[..., 1] = y
        corrected_points[..., 2] = z

    elif axis == 1:
        x = points[..., 0] * np.cos(angle) + points[..., 2] * np.sin(angle)
        z = points[..., 2] * np.cos(angle) - points[..., 0] * np.sin(angle)
        corrected_points[..., 0] = x
        corrected_points[..., 2] = z

    elif axis == 2:
        x = points[..., 0] * np.cos(angle) - points[..., 1] * np.sin(angle)
        y = points[..., 0] * np.sin(angle) + points[..., 1] * np.cos(angle)
        corrected_points[..., 0] = x
        corrected_points[..., 1] = y
    else:
        raise AttributeError(f'Axis {axis} not supported.')

    return corrected_points
