#
# @author  Daniel Maturana
# @year    2015
#
# @attention Copyright (c) 2015
# @attention Carnegie Mellon University
# @attention All rights reserved.
#
# @=



import math
import numpy as np
from rospy_geomutils import Quaternion

from tf import transformations

q = Quaternion.from_roll_pitch_yaw(0, 0, 2 * math.pi / 16)
v = [ 1, 0, 0 ]
print(v)
for i in range (16):
    v = q.rotate(v)
    print(v)

qinv = q.inverse()
q3 = (qinv * q)
#print(np.allclose( q3, Quaternion.identity()))

qa = transformations.quaternion_about_axis( 0.123, (1,0,0) )
qb = Quaternion.from_angle_axis(0.123, (1,0,0))
print
print(qb.w, qb.x, qb.y, qb.z)
print(qb)



