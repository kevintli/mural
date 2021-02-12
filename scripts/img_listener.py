#!/usr/bin/env python
# PKG = 'numpy_tutorial'
# import roslib; roslib.load_manifest(PKG)
from std_msgs.msg import Int32MultiArray
import rospy
import numpy as np

def callback(data):
    image = np.array(data.data).reshape((32, 32, 3))
    print rospy.get_name(), "I heard %s"%str(image)

def listener():
    rospy.init_node('listener')
    rospy.Subscriber("images", Int32MultiArray, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
