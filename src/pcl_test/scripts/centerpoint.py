#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import math

CENTER_X = 320/2
CENTER_Y = 240/2
window_size = 10
bridge = CvBridge()

def depth_callback(data):
    try:
        avg_depth = 0
        count = 0
        cv_image = bridge.imgmsg_to_cv2(data)
        for i in range(window_size):
            for j in range(window_size):
                point = cv_image[CENTER_X+i, CENTER_Y+j]
                if math.isnan(point) or point > 4.0: 
                    continue
                else:
                    count += 1
                    avg_depth += cv_image[CENTER_X + i, CENTER_Y + j]

        if count != 0:
            print avg_depth / float(count) 
        else:
            print 'no points in window'
    except CvBridgeError as e:
        print e


if __name__ == '__main__':
    rospy.init_node('centerpoint_node')
    image_sub = rospy.Subscriber('/guidance/real_depth_image', Image, depth_callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print 'Shutting down'
