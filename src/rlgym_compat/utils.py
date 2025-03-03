import numpy as np
from rlbot.flat import BoxShape, Rotator, Vector3


def create_default_init(slots):
    func = "def __init__(self):"
    for attr in slots:
        func += " self.{}=None;".format(attr)
    return func


def vector_to_numpy(vector: Vector3):
    return np.asarray([vector.x, vector.y, vector.z])


def rotator_to_numpy(rotator: Rotator):
    return np.asarray([rotator.pitch, rotator.yaw, rotator.roll])


def write_vector_into_numpy(np_vec: np.ndarray, vec: Vector3):
    np_vec[0] = vec.x
    np_vec[1] = vec.y
    np_vec[2] = vec.z


def compare_hitbox_shape(shape: BoxShape, length, width, height):
    return shape.length == length and shape.width == width and shape.height == height
