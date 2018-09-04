#
# @author  Daniel Maturana
# @year    2015-2018
#
# @=

import numpy as np
from tf import transformations as tftf

from geometry_msgs.msg import Quaternion as QuaternionMsg


class Quaternion(object):
    """ Quaternion.
    Quaternions are ordered as xyzw, since we use ROS tf.transformations.py.
    Note that Eigen ctor and Censi geometry use xyzw.
    """
    def __init__(self, *args):
        if len(args) == 0:
            self.q = np.array((0.0, 0.0, 0.0, 1.0))
        elif len(args) == 4:
            self.q = np.asarray(args[:])
        elif len(args) == 1:
            if isinstance(args[0], Quaternion):
                self.q = args[0].q.copy()
            elif len(args[0]) == 4:
                self.q = np.array(args[0][:])
        else:
            raise TypeError("invalid initializer")
        norm = np.sqrt(np.dot(self.q, self.q))
        assert np.isclose(norm, 1.0)

    def __mul__(self, other):
        a = self.q
        b = other.q
        return tftf.quaternion_multiply(a, b)

    def rotate_vec(self, v):
        R33 = self.to_array33()
        return np.dot(R33, v)

    @staticmethod
    def identity():
        return Quaternion()

    @property
    def w(self):
        return self.q[3]

    @property
    def x(self):
        return self.q[0]

    @property
    def y(self):
        return self.q[1]

    @property
    def z(self):
        return self.q[2]

    def __getitem__(self, i):
        return self.q[i]

    def __repr__(self):
        return ('Quaternion(x=%f, y=%f, z=%f, w=%f)' %
               (self.q[0], self.q[1], self.q[2], self.q[3]))

    def inverse(self):
        return Quaternion(-self.q[0], -self.q[1], -self.q[2], self.q[3])

    @staticmethod
    def from_roll_pitch_yaw(roll, pitch, yaw):
        # note: convention differs from older from_roll_pitch_yaw,
        # but this is what ROS seems to prefer (e.g. when using static_transform_publisher).
        # sxyz is 'static frame x-y-z'
        return Quaternion(tftf.quaternion_from_euler(roll, pitch, yaw, 'sxyz'))

    @staticmethod
    def from_angle_axis(theta, axis):
        x, y, z = axis
        norm = np.sqrt(x*x + y*y + z*z)
        if np.isclose(norm, 0.):
            return Quaternion.identity()
        t = np.sin(theta/2)/norm
        return Quaternion(x*t, y*t, z*t, np.cos(theta/2.))

    @staticmethod
    def from_quaternion_msg(msg):
        """ for use with ROS geometry_msgs """
        return Quaternion(msg.x, msg.y, msg.z, msg.w)

    def to_roll_pitch_yaw(self):
        return tftf.euler_from_quaternion(self.to_xyzw(), 'sxyz')

    def to_angle_axis(self):
        halftheta = np.arccos(self.w)
        if abs(halftheta) < 1e-12:
            return 0., np.array((0., 0., 1.))
        else:
            theta = halftheta * 2
            axis = np.array(self.q[0:3]) / np.sin(halftheta)
            return theta, axis

    def to_array44(self):
        return tftf.quaternion_matrix(self.to_xyzw())

    def to_array33(self):
        return tftf.quaternion_matrix(self.to_xyzw())[:3,:3]

    @staticmethod
    def from_array33(R, isprecise=False):
        R44 = np.eye(4)
        R44[:3, :3] = R
        return Quaternion(tftf.quaternion_from_matrix(R44))

    @staticmethod
    def from_array44(R, isprecise=False):
        return Quaternion(tftf.quaternion_from_matrix(R))

    def to_wxyz(self):
        return np.array((self.q[3], self.q[0], self.q[1], self.q[2]))

    def to_xyzw(self):
        return self.q[:]

    def to_quaternion_msg(self):
        qmsg = QuaternionMsg()
        qmsg.x = self.q[0]
        qmsg.y = self.q[1]
        qmsg.z = self.q[2]
        qmsg.w = self.q[3]
        return qmsg

    def interpolate(self, other, this_weight):
        q0 = self.to_xyzw()
        q1 = other.to_xyzw()
        q = tftf.quaternion_slerp(q0, q1, 1.0 - this_weight)
        return Quaternion(q)

    @staticmethod
    def make_random():
        return Quaternion(tftf.random_quaternion())

    def isclose(self, other):
        if np.sign(self.w) != np.sign(other.w):
            return np.allclose(self.q, -other.q)
        return np.allclose(self.q, other.q)


if __name__ == "__main__":
    # q = Quaternion.make_random()
    rpy = (1., 2., 3.)
    q1 = Quaternion.from_roll_pitch_yaw(*rpy)
    q2 = Quaternion.from_roll_pitch_yaw2(*rpy)
    print q1, q2
