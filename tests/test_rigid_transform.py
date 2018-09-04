import random
import numpy as np

from rospy_geomutils import Quaternion
from rospy_geomutils import RigidTransform


"""
q = Quaternion([1, 0, 0, 0])
t = [ 1, 2, 3 ]
m = RigidTransform(q, t)
print "m"
print m.to_homogeneous_matrix()

q2 = Quaternion.from_roll_pitch_yaw(np.pi / 4, 0, 0)
t2 = [ 0, 0, 0 ]
m2 = RigidTransform(q2, t2)
print "m2"
print m2.to_homogeneous_matrix()

m3 = m * m2
print "m * m2"
print m3.to_homogeneous_matrix()

m4 = m2 * m
print "m * m2"
print m4
"""

random.seed(28)


def make_random_transform():
    q = Quaternion.make_random()
    translation = [random.uniform(-100, 100),
                   random.uniform(-100, 100),
                   random.uniform(-100, 100)]
    return RigidTransform(q, translation)


def test_inverse():
    identity = np.identity(4)
    for _ in range(100):
        # generate a bunch of random rigid body transforms, then compose them
        # and apply their inverses
        # the result should be the identity transform
        num_transforms = random.randint(0, 10)
        ms = [make_random_transform() for _ in range(num_transforms)]
        inverses = [m.inverse() for m in ms]
        inverses.reverse()
        r = RigidTransform.identity()
        for m in ms + inverses:
            r *= m
        errs = (identity - r.to_array44()).flatten().tolist()[0]
        sse = np.dot(errs, errs)
        assert(np.isclose(sse, 0.))


def test_composition():
    t = RigidTransform.identity()
    m = np.identity(4)
    for _ in range(1000):
        n = make_random_transform()
        t = t * n
        m = np.dot(m, n.to_array44())
        errs = (t.to_array44() - m).flatten().tolist()[0]
        sse = np.dot(errs, errs)
        assert(np.isclose(sse, 0.))
