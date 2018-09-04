#
# @author  Daniel Maturana
# @year    2015
#
# @attention Copyright (c) 2015
# @attention Carnegie Mellon University
# @attention All rights reserved.
#
# @=



import numpy as np
import rospy_geomutils

import geometry
from tf import transformations

rt = rospy_geomutils.RigidTransform.identity()

q1 = transformations.quaternion_about_axis( 0.123, (1, 0, 0) )
q2 = geometry.quaternion_from_axis_angle( np.array((1., 0., 0.)), 0.123 )
q3 = rospy_geomutils.Quaternion.from_angle_axis( 0.123, (1, 0, 0) )

