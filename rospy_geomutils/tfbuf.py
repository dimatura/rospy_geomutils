import copy

import rospy
import tf2_ros
import rosbag
import image_geometry


def get_pinhole_camera_model(bag_fname, infotopic):
    """ assuming model is static """
    bag = rosbag.Bag(bag_fname, 'r')
    for topic, msg, bts in bag.read_messages(infotopic):
        cam = image_geometry.PinholeCameraModel()
        cam.fromCameraInfo(msg)
        break
    return cam


def get_tfbuf(bag, duration=8000000):
    """ given a bag, load all the transforms into a tf buffer. """
    tfbuf = tf2_ros.BufferCore(rospy.Duration(duration))
    for topic, msg, ts in bag.read_messages('/tf'):
        for tfi in msg.transforms:
            tfbuf.set_transform(copy.deepcopy(tfi), 'default')
    return tfbuf


def get_tfbuf_bagfiles(bag_fnames, duration=8000000):
    tfbuf = tf2_ros.BufferCore(rospy.Duration(duration))
    for bag_fname in bag_fnames:
        with rosbag.Bag(bag_fname, 'r') as bag:
            for topic, msg, ts in bag.read_messages('/tf'):
                for tfi in msg.transforms:
                    tfbuf.set_transform(copy.deepcopy(tfi), 'default')
    return tfbuf


def get_info(bag):
    _, topic_info = bag.get_type_and_topic_info()
    return topic_info


def GenImageMsgsWithTransforms(bag_fname, img_topic, parent_id, child_id):
    bag = rosbag.Bag(bag_fname, 'r')
    tfbuf = get_tfbuf(bag)
    for topic, msg, bts in bag.read_messages([img_topic]):
        ts = msg.header.stamp
        try:
            tfi = tfbuf.lookup_transform_core(parent_id, child_id, ts)
        except tf2_ros.ExtrapolationException:
            print('extrapolation exception at {}'.format(ts))
            continue
        yield (msg, tfi)
