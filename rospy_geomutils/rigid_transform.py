# @author  Daniel Maturana
# @year    2015-2018
#
# @attention Copyright (c) 2015
# @attention Carnegie Mellon University
# @attention All rights reserved.
#
# @=

import numpy as np
from tf import transformations as tftf

from .quaternion import Quaternion

from geometry_msgs.msg import Pose as PoseMsg
from geometry_msgs.msg import Transform as TransformMsg

# from geometry_msgs.msg import Point as PointMsg
# from geometry_msgs.msg import Vector3 as Vector3Msg

class RigidTransform(object):
    def __init__(self, rotation_quat=None, translation_vec=None):
        if rotation_quat is not None:
            self.quat = Quaternion(rotation_quat)
        else:
            self.quat = Quaternion.identity()
        if translation_vec is not None:
            self.tvec = np.asarray(translation_vec)
        else:
            self.tvec = np.zeros(3)

    def inverse(self):
        """ returns inverse of this RigidTransform
        """
        qinv = self.quat.inverse()
        return RigidTransform(qinv, qinv.rotate_vec(-self.tvec))

    def interpolate(self, other_transform, this_weight):
        assert this_weight >= 0 and this_weight <= 1
        t = self.tvec * this_weight + other_transform.tvec * (1 - this_weight)
        r = self.quat.interpolate(other_transform.quat, this_weight)
        return RigidTransform(r, t)

    def __repr__(self):
        return 'RigidTransform(%r, %r)' % (self.quat, self.tvec)

    def __mul__(self, other):
        if isinstance(other, RigidTransform):
            t = self.quat.rotate_vec(other.tvec) + self.tvec
            q = self.quat * other.quat
            return RigidTransform(q, t)
        else:
            olen = len(other)
            if olen == 3:
                r = np.array(self.quat.rotate_vec(other))
                return r + self.tvec
            elif olen == 4:
                return np.dot(self.to_array44(), other)
            else:
                raise ValueError("operand must be RigidTransform, "
                                 "3-vector or 4-vector")

    def to_array44(self):
        result = self.quat.to_array44()
        result[:3, 3] = self.tvec
        return result

    def to_roll_pitch_yaw_xyz(self):
        r, p, y = self.quat.to_roll_pitch_yaw()
        return np.array((r, p, y, self.tvec[0], self.tvec[1], self.tvec[2]))

    @staticmethod
    def from_roll_pitch_yaw_xyz(r, p, yaw, x, y, z):
        q = Quaternion.from_roll_pitch_yaw(r, p, yaw)
        return RigidTransform(q, (x, y, z))

    @staticmethod
    def from_roll_pitch_yaw(r, p, yaw):
        q = Quaternion.from_roll_pitch_yaw(r, p, yaw)
        return RigidTransform(q, (0., 0., 0.))

    @staticmethod
    def from_pose_msg(msg):
        """ for use with ROS geometry_msgs/Pose """
        q = (msg.orientation.x,
             msg.orientation.y,
             msg.orientation.z,
             msg.orientation.w)
        t = (msg.position.x, msg.position.y, msg.position.z)
        return RigidTransform(q, t)

    @staticmethod
    def from_transform_msg(msg):
        """ for use with ROS geometry_msgs/Tranform """
        q = (msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w)
        t = (msg.translation.x, msg.translation.y, msg.translation.z)
        return RigidTransform(q, t)

    @staticmethod
    def from_quaternion_msg(msg):
        q = (msg.x, msg.y, msg.z, msg.w)
        return RigidTransform(q, (0., 0., 0.))

    @staticmethod
    def from_point_msg(msg):
        t = (msg.x, msg.y, msg.z)
        return RigidTransform((0., 0., 0., 1.), t)

    def to_pose_msg(self):
        msg = PoseMsg()
        (msg.orientation.x,
         msg.orientation.y,
         msg.orientation.z,
         msg.orientation.w) = self.quat.to_xyzw()
        (msg.position.x, msg.position.y, msg.position.z) = self.tvec
        return msg

    def to_transform_msg(self):
        msg = TransformMsg()
        (msg.rotation.x,
         msg.rotation.y,
         msg.rotation.z,
         msg.rotation.w) = self.quat.to_xyzw()
        (msg.translation.x,
         msg.translation.y,
         msg.translation.z) = self.tvec
        return msg

    @property
    def quaternion(self):
        return self.quat

    @property
    def translation(self):
        return self.tvec

    @property
    def tx(self):
        return self.tvec[0]

    @property
    def ty(self):
        return self.tvec[1]

    @property
    def tz(self):
        return self.tvec[2]

    @property
    def qw(self):
        return self.quat.w

    @property
    def qx(self):
        return self.quat.x

    @property
    def qy(self):
        return self.quat.y

    @property
    def qz(self):
        return self.quat.z

    @staticmethod
    def identity():
        q = Quaternion.identity()
        tr = np.zeros(3)
        return RigidTransform(q, tr)
