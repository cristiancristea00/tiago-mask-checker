from sensor_msgs.msg import CompressedImage
from utils import *
import warnings
import argparse
import rospy
import sys

warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('-t', '--tracker', type = str, default = 'CSRT',
				help = 'Tracker type: BOOSTING/MIL/KCF/TLD/MEDIANFLOW/MOSSE/CSRT')
ap.add_argument('-c', '--confidence', type = float, default = 0.5,
				help = 'Minimum probability to filter weak detections')
ap.add_argument('-T', '--threshold', type = int, default = 60,
				help = 'Minimum distance between face detection and tracker')
ap.add_argument('-v', '--value', type = int, default = 5,
				help = 'Number of frames between tracker and detector sync')
ap.add_argument('-w', '--wait', type = int, default = 20,
				help = 'Number of frames to wait before starting tracker after a face is detected')
ap.add_argument('-s', '--state', type = int, default = 15,
				help = 'Number of frames to ait before a message is displayed')
args = vars(ap.parse_args())

normal_wrapper = AtomicWrapper()
temp_wrapper = AtomicWrapper()
thermal_wrapper = AtomicWrapper()


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
	locations, predictions = None, None

	tracker = Tracker(args['tracker'])
	face_and_mask_detector = FaceAndMaskDetector(args['confidence'])
	temp_checker = TemperatureChecker()
	person_waiter = WaitingForPerson(tracker, face_and_mask_detector, args['value'], args['wait'], args['threshold'])
	person_checker = CheckingPerson(tracker, face_and_mask_detector, args['value'], args['wait'], args['threshold'],
									args['state'])

	while True:
		# Get current images
		normal_wrapper.set(normal)
		curr_normal = normal_wrapper.get()
		temp_wrapper.set(temp)
		curr_temp = temp_wrapper.get()
		thermal_wrapper.set(thermal)
		curr_thermal = thermal_wrapper.get()

		if current_state == 'waiting':
			locations, predictions = person_waiter.run_prediction(curr_normal)

		if person_waiter.person_detected:
			current_state = 'person_detected'

		if current_state == 'person_detected':
			locations, predictions = person_checker.run_prediction(curr_normal)
			curr_normal = draw_boxes(locations, predictions, curr_normal)

		curr_normal = draw_tracker(tracker.track_ok, curr_normal, locations, args['tracker'])

		frame = np.hstack((curr_normal, curr_thermal))

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
	rospy.spin()


if __name__ == '__main__':
	listener()
