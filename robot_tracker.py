from sensor_msgs.msg import CompressedImage
from utils import *
import warnings
import argparse
import rospy
import os

warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('-t', '--tracker', type = str, default = 'CSRT',
				help = 'Tracker type: BOOSTING/MIL/KCF/TLD/MEDIANFLOW/MOSSE/CSRT')
ap.add_argument('-c', '--confidence', type = float, default = 0.5,
				help = 'Minimum probability to filter weak detections')
ap.add_argument('-T', '--threshold', type = int, default = 60,
				help = 'Minimum distance between face detection and tracker')
ap.add_argument('-v', '--value', type = int, default = 5, help = 'Number of frames between tracker and detector sync')
ap.add_argument('-w', '--wait', type = int, default = 20,
				help = 'Number of frames to wait before starting tracker after a face is detected')
args = vars(ap.parse_args())

normal_wrapper = AtomicWrapper()
temp_wrapper = AtomicWrapper()
thermal_wrapper = AtomicWrapper()


def callback_normal(data):
	global normal
	normal_wrapper.set(cv.imdecode(np.fromstring(data.data, np.uint8), cv.IMREAD_COLOR))
	normal = normal_wrapper.get()[120:355, 150:500]


def callback_thermal(data):
	global thermal
	thermal_wrapper.set(cv.imdecode(np.fromstring(data.data, np.uint8), cv.IMREAD_COLOR))
	thermal = cv.resize(thermal_wrapper.get(), (350, 235), interpolation = cv.INTER_NEAREST)


def callback_temp(data):
	global temp
	temp_wrapper.set(cv.imdecode(np.fromstring(data.data, np.uint8), cv.IMREAD_ANYDEPTH))
	temp = cv.resize(temp_wrapper.get(), (350, 235), interpolation = cv.INTER_NEAREST)


def video():
	tracker = create_tracker(args['tracker'])
	tracker_started = False
	track_ok = False
	counter = args['value']
	wait_counter = args['wait']

	# Temp list
	temps = []

	global thermal
	global normal
	global temp

	while True:

		# Get current images
		normal_wrapper.set(normal)
		curr_normal = normal_wrapper.get()
		temp_wrapper.set(temp)
		curr_temp = temp_wrapper.get()
		thermal_wrapper.set(thermal)
		curr_thermal = thermal_wrapper.get()

		# Detect faces in the frame and determine if they are wearing a face mask or not
		(locations, predictions) = detect_and_predict_mask(curr_normal, face_net, mask_net, args['confidence'])

		# Loop over the detected face locations and their corresponding  locations
		for (box, predic) in zip(locations, predictions):
			# Unpack the bounding box and predictions
			(start_x, start_y, end_x, end_y) = box
			(with_mask, with_mask_no_nose, with_mask_under, without_mask) = predic

			# Determine the class label and color we'll use to draw  the bounding box and text
			max_prob = max(with_mask, with_mask_no_nose, with_mask_under, without_mask)
			if max_prob == with_mask:
				label = 'Mask is OK'
				color = (0, 255, 0)  # Green
			elif max_prob == with_mask_no_nose:
				label = 'Cover your nose'
				color = (15, 219, 250)  # Yellow
			elif max_prob == with_mask_under:
				label = 'Cover yourself'  # Orange
				color = (0, 104, 240)
			else:
				label = 'NO mask'
				color = (0, 0, 255)  # Red

			# Include the probability in the label
			label = '{}: {:.2f}%'.format(label, max_prob * 100)

			# Display the label and bounding box rectangle on the output frame
			cv.putText(curr_normal, label, (start_x, start_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv.rectangle(curr_normal, (start_x, start_y), (end_x, end_y), color, 2)
			cv.rectangle(curr_thermal, (start_x, start_y), (end_x, end_y), color, 2)

			# Print the temperature
			try:
				t = np.mean(curr_temp[start_y + 15:end_y - 35, start_x + 14:end_x - 14] - 1000) / 10.0
			except ValueError:
				t = 0
			if t > 34:
				temps.append(t)

		# Decrement the wait counter for every frame where a face is detected
		if len(locations) != 0 and wait_counter != 0:
			wait_counter -= 1

		# Start tracker on the first face detected
		if not tracker_started and len(locations) != 0 and wait_counter == 0:
			tracker_started = True
			bbox = convert_2points_to_1point_and_dims(locations[0])
			track_ok = tracker.init(curr_normal, bbox)

		# Update tracker
		track_ok, bbox = tracker.update(curr_normal)

		# For every frame decrease the counter
		if counter != 0:
			counter -= 1

		# For every set frames, reposition the tracker on the detected face if the distance between their centers are
		# under the threshold value
		if tracker_started and counter == 0 and track_ok:
			counter = args['value']
			tracker_center = get_middle_point(convert_1point_and_dims_to_2points(bbox))
			for box in locations:
				(start_x, start_y, end_x, end_y) = box
				detector_center = get_middle_point(((start_x, start_y), (end_x, end_y)))
				if dist(tracker_center, detector_center) <= args['threshold']:
					bbox = convert_2points_to_1point_and_dims((start_x, start_y, end_x, end_y))
					tracker = create_tracker(args['tracker'])
					track_ok = tracker.init(curr_normal, bbox)
					break

		# Draw bounding box
		if track_ok:
			# Tracking success
			points = convert_1point_and_dims_to_2points(bbox)
			cv.rectangle(curr_normal, points[0], points[1], (232, 189, 19), 2, 1)
		else:
			# Tracking failure
			cv.putText(curr_normal, 'Tracking failure detected!', (5, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

		# Display tracker type on frame
		cv.putText(curr_normal, args['tracker'] + ' Tracker', (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50, 170, 50), 2)

		frame = np.concatenate((curr_normal, curr_thermal), axis = 1)

		# Display the result
		cv.imshow('Video stream', frame)

		# Exit if Q pressed
		key = cv.waitKey(1) & 0xFF
		if key == ord('q'):
			break

	cv.destroyAllWindows()
	os._exit(0)


def listener():
	rospy.init_node('listener', anonymous = True)
	rospy.Subscriber('/xtion/rgb/image_raw/compressed', CompressedImage, callback_normal)
	rospy.Subscriber('/optris/thermal_image/compressed', CompressedImage, callback_temp)
	rospy.Subscriber('/optris/thermal_image_view/compressed', CompressedImage, callback_thermal)
	video()
	rospy.spin()


if __name__ == '__main__':
	listener()
