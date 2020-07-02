from sensor_msgs.msg import CompressedImage
from threading import Thread
from time import sleep
import imutils
import numpy as np
import rospy
import cv2
import sys
import os

global thermal
global normal


def callback_normal(data):
	global normal
	normal = cv2.imdecode(np.fromstring(data.data, np.uint8), cv2.IMREAD_COLOR)
	normal = cv2.resize(normal, (800, 600))

def callback_thermal(data):
	global thermal
	thermal = cv2.imdecode(np.fromstring(data.data, np.uint8), cv2.IMREAD_COLOR)
	thermal = cv2.resize(thermal, (800, 600))

def video():
	global thermal
	global normal
	
	while True:
		# Read a new frame if possible
		try:
			frame = np.concatenate((normal, thermal), axis=1)
		except ValueError:
			pass
		except NameError:
			pass
		
		# Display result
		try:
			cv2.imshow('Video stream', frame)
		except UnboundLocalError:
			pass
		key = cv2.waitKey(1) & 0xFF
		
		# Exit if Q pressed
		if key == ord('q'):
			break

	cv2.destroyAllWindows()
	os._exit(1)
	
	

def listener():
	rospy.init_node('listener', anonymous=True)
	merger = Thread(target = video)
	rospy.Subscriber('/xtion/rgb/image_raw/compressed', CompressedImage, callback_normal)
	rospy.Subscriber('/optris/thermal_image_view/compressed', CompressedImage, callback_thermal)
	merger.start()
	rospy.spin()

if __name__ == '__main__':
	listener()

