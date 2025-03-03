from dataclasses import dataclass
from typing import Optional, Self

import numpy as np
from rlbot.flat import Physics

from .math import (
    euler_to_rotation,
    quat_to_euler,
    quat_to_rot_mtx,
    rotation_to_quaternion,
)
from .utils import create_default_init, write_vector_into_numpy


@dataclass(init=False)
class PhysicsObject:
    INV_VEC = np.array([-1, -1, 1], dtype=np.float32)
    INV_MTX = np.array([[-1, -1, -1], [-1, -1, -1], [1, 1, 1]], dtype=np.float32)

    position: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    _quaternion: Optional[np.ndarray]
    _rotation_mtx: Optional[np.ndarray]
    _euler_angles: Optional[np.ndarray]

    _rlbot_euler_angles: np.ndarray

    __slots__ = tuple(__annotations__)

    exec(create_default_init(__slots__))

    def inverted(self) -> Self:
        inv = PhysicsObject()
        inv.position = self.position * PhysicsObject.INV_VEC
        inv.linear_velocity = self.linear_velocity * PhysicsObject.INV_VEC
        inv.angular_velocity = self.angular_velocity * PhysicsObject.INV_VEC
        if (
            self._rotation_mtx is not None
            or self._quaternion is not None
            or self._euler_angles is not None
        ):
            inv.rotation_mtx = self.rotation_mtx * PhysicsObject.INV_MTX
        return inv

    @property
    def quaternion(self) -> np.ndarray:
        if self._quaternion is None:
            if self._rotation_mtx is not None:
                self._quaternion = rotation_to_quaternion(self._rotation_mtx)
            elif self._euler_angles is not None:
                self._rotation_mtx = euler_to_rotation(self.euler_angles)
                self._quaternion = rotation_to_quaternion(self._rotation_mtx)
            else:
                raise ValueError
        return self._quaternion

    @quaternion.setter
    def quaternion(self, val: np.ndarray):
        self._quaternion = val
        self._rotation_mtx = None
        self._euler_angles = None

    @property
    def rotation_mtx(self) -> np.ndarray:
        if self._rotation_mtx is None:
            if self._quaternion is not None:
                self._rotation_mtx = quat_to_rot_mtx(self._quaternion)
            elif self._euler_angles is not None:
                self._rotation_mtx = euler_to_rotation(self._euler_angles)
            else:
                raise ValueError
        return self._rotation_mtx

    @rotation_mtx.setter
    def rotation_mtx(self, val: np.ndarray):
        self._rotation_mtx = val
        self._quaternion = None
        self._euler_angles = None

    @property
    def euler_angles(self) -> np.ndarray:
        if self._euler_angles is None:
            if self._quaternion is not None:
                self._euler_angles = quat_to_euler(self._quaternion)
            elif self._rotation_mtx is not None:
                self._quaternion = rotation_to_quaternion(self._rotation_mtx)
                self._euler_angles = quat_to_euler(self._quaternion)
            else:
                raise ValueError
        return self._euler_angles

    @euler_angles.setter
    def euler_angles(self, val: np.ndarray):
        self._euler_angles = val
        self._quaternion = None
        self._rotation_mtx = None

    @property
    def forward(self) -> np.ndarray:
        return self.rotation_mtx[:, 0]

    @property
    def right(self) -> np.ndarray:
        return self.rotation_mtx[:, 1]

    @property
    def left(self) -> np.ndarray:
        return self.rotation_mtx[:, 1] * -1

    @property
    def up(self) -> np.ndarray:
        return self.rotation_mtx[:, 2]

    @property
    def pitch(self) -> float:
        return self.euler_angles[0]

    @property
    def yaw(self) -> float:
        return self.euler_angles[1]

    @property
    def roll(self) -> float:
        return self.euler_angles[2]

    @staticmethod
    def create_compat_physics_object():
        physics_object = PhysicsObject()
        physics_object.position = np.zeros(3)
        physics_object.linear_velocity = np.zeros(3)
        physics_object.angular_velocity = np.zeros(3)
        physics_object._rlbot_euler_angles = np.zeros(3)
        return physics_object

    def update(self, physics: Physics):
        write_vector_into_numpy(self.position, physics.location)
        write_vector_into_numpy(self.linear_velocity, physics.velocity)
        write_vector_into_numpy(self.angular_velocity, physics.angular_velocity)
        # Need to do it like this so that the property setter gets called
        self._rlbot_euler_angles[0] = physics.rotation.pitch
        self._rlbot_euler_angles[1] = physics.rotation.yaw
        self._rlbot_euler_angles[2] = physics.rotation.roll
        self.euler_angles = self._rlbot_euler_angles
