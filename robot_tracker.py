from sensor_msgs.msg import CompressedImage
from utils import *
import argparse
import rospy
import sys

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-t', '--tracker', type = str, default = 'CSRT',
						help = 'Tracker type: BOOSTING/MIL/KCF/TLD/MEDIANFLOW/MOSSE/CSRT')
arg_parser.add_argument('-c', '--confidence', type = float, default = 0.5,
						help = 'Minimum probability to filter weak detections')
arg_parser.add_argument('-T', '--threshold', type = int, default = 60,
						help = 'Minimum distance between face detection and tracker')
arg_parser.add_argument('-v', '--value', type = int, default = 5,
						help = 'Number of frames between tracker and detector sync')
arg_parser.add_argument('-w', '--wait', type = int, default = 20,
						help = 'Number of frames to wait before starting tracker after a face is detected')
arg_parser.add_argument('-s', '--state', type = int, default = 15,
						help = 'Number of frames to ait before a message is displayed')
args = vars(arg_parser.parse_args())

normal_wrapper = AtomicWrapper()
temp_wrapper = AtomicWrapper()
thermal_wrapper = AtomicWrapper()

global thermal
global normal
global temp


def callback_normal(data):
	normal_wrapper.set(cv.imdecode(np.fromstring(data.data, np.uint8), cv.IMREAD_COLOR))
	global normal
	normal = normal_wrapper.get()[120:355, 150:500]


def callback_thermal(data):
	thermal_wrapper.set(cv.imdecode(np.fromstring(data.data, np.uint8), cv.IMREAD_COLOR))
	global thermal
	thermal = cv.resize(thermal_wrapper.get(), (350, 235), interpolation = cv.INTER_NEAREST)


def callback_temp(data):
	temp_wrapper.set(cv.imdecode(np.fromstring(data.data, np.uint8), cv.IMREAD_ANYDEPTH))
	global temp
	temp = cv.resize(temp_wrapper.get(), (350, 235), interpolation = cv.INTER_NEAREST)


def video():
	global thermal
	global normal
	global temp

	current_state = 'waiting'

	tracker = Tracker(args['tracker'])
	detector = FaceAndMaskDetector(args['confidence'])
	temp_checker = TemperatureChecker()
	person_waiter = WaitingForPerson(tracker, detector, args['value'], args['wait'], args['threshold'])
	person_checker = CheckingPerson(tracker, detector, args['value'], args['wait'], args['threshold'], args['state'])

	while True:
		# Get current images
		normal_wrapper.set(normal)
		curr_normal = normal_wrapper.get()
		temp_wrapper.set(temp)
		curr_temp = temp_wrapper.get()
		thermal_wrapper.set(thermal)
		curr_thermal = thermal_wrapper.get()

		if current_state == 'waiting':
			person_waiter.run_prediction(curr_normal)

		if person_waiter.person_in_frame():
			current_state = 'person_detected'

		if current_state == 'person_detected':
			person_checker.run_prediction(curr_normal)

		frame = np.vstack((curr_normal, curr_thermal))

		# Display the result
		cv.imshow('Video stream', frame)

		# Exit if Q pressed
		if cv.waitKey(1) & 0xFF == ord('q'):
			break

	cv.destroyAllWindows()
	sys.exit(0)


def listener():
	rospy.init_node('listener', anonymous = True)
	rospy.Subscriber('/xtion/rgb/image_raw/compressed', CompressedImage, callback_normal)
	rospy.Subscriber('/optris/thermal_image/compressed', CompressedImage, callback_temp)
	rospy.Subscriber('/optris/thermal_image_view/compressed', CompressedImage, callback_thermal)
	video()


if __name__ == '__main__':
	listener()
