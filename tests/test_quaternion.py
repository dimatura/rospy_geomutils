
import math
import random
import numpy as np

from rospy_geomutils import Quaternion

random.seed(28)

"""
q = Quaternion.from_roll_pitch_yaw (0, 0, 2 * math.pi / 16)
v = [ 1, 0, 0 ]
print v
for i in range (16):
    v = q.rotate (v)
    print v
"""


def mod2pi_positive(vin):
    """ assumes vin is positive. """
    q = vin / (2*np.pi) + 0.5
    qi = int(q)
    return vin - qi*2*np.pi


def mod2pi(vin):
    """ modulo 2pi, supports negative vin """
    if (vin < 0):
        return -mod2pi_positive(-vin)
    return mod2pi_positive(vin)


def mod2pi_ref(ref, vin):
    return ref + mod2pi(vin - ref)


def test_toarray():
    q = Quaternion()
    assert(np.allclose(q.to_wxyz(), [1., 0., 0., 0.]))
    assert(np.allclose(q.to_xyzw(), [0., 0., 0., 1.]))


def test_msg():
    q = Quaternion.make_random()
    msg = q.to_quaternion_msg()
    for m in 'wxyz':
        assert(np.isclose(getattr(q, m), getattr(msg, m)))
    q2 = Quaternion.from_quaternion_msg(msg)
    assert q.isclose(q2)


def test_array():
    q1 = Quaternion.make_random()
    R1 = q1.to_array44()
    q2 = Quaternion.from_array44(R1)
    assert q1.isclose(q2)


def test_rpy():
    q1 = Quaternion.make_random()
    rpy1 = q1.to_roll_pitch_yaw()
    q2 = Quaternion.from_roll_pitch_yaw(*rpy1)
    assert(q1.isclose(q2))


def test_interpolation():
    q = Quaternion.from_roll_pitch_yaw(0, 0, 2 * math.pi / 16)
    q2 = Quaternion.from_roll_pitch_yaw(0, 0, 0)
    rpy_start = np.array(q.to_roll_pitch_yaw())
    rpy_goal = np.array(q2.to_roll_pitch_yaw())
    for i in range(101):
        alpha = i / 100.
        qinterp = q2.interpolate(q, alpha)
        rpy_interp = np.array(qinterp.to_roll_pitch_yaw())
        rpy_expected = (rpy_goal * alpha + rpy_start * (1 - alpha))
        err = rpy_expected - rpy_interp
        for k in [0, 1, 2]:
            assert(np.isclose(err[k], 0.))


def test_angle_axis():
    for _ in range(100):
        theta = random.uniform(-np.pi, np.pi)
        axis = np.array([random.random(), random.random(), random.random()])
        axis /= np.linalg.norm(axis)
        q = Quaternion.from_angle_axis(theta, axis)
        theta_check, axis_check = q.to_angle_axis()
        if np.dot(axis, axis_check) < 0:
            theta_check *= -1
            axis_check *= -1
        theta_check = mod2pi_ref(theta, theta_check)
        dtheta = theta_check - theta
        daxis = axis - axis_check
        assert np.isclose(dtheta, 0.)
        assert np.isclose(np.linalg.norm(daxis), 0.)
