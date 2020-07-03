from tensorflow.keras.models import load_model
from sensor_msgs.msg import CompressedImage
from utils import *
import argparse
import rospy
import os

ap = argparse.ArgumentParser()
ap.add_argument('-t', '--tracker', type=str, default='CSRT', help='Tracker type: BOOSTING/MIL/KCF/TLD/MEDIANFLOW/MOSSE/CSRT')
ap.add_argument('-c', '--confidence', type=float, default=0.5, help='Minimum probability to filter weak detections')
ap.add_argument('-T', '--threshold', type=int, default=60, help='Minimum distance between face detection and tracker')
ap.add_argument('-v', '--value', type=int, default=5, help='Number of frames between tracker and detector sync')
ap.add_argument('-w', '--wait', type=int, default=20, help='Number of frames to wait before starting tracker after a face is detected')
args = vars(ap.parse_args())

# Load our serialized face detector model from disk
print('[INFO] Loading face detector model...')
face_net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Load the face mask detector model from disk
print('[INFO] Loading face mask detector model...')
mask_net = load_model('mask_detector.model')

# Init global variables
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
	tracker = create_tracker(args['tracker'])
	tracker_started = False
	counter = args['value']
	wait_counter = args['wait']

	global thermal
	global normal
	while True:

		try:
			# Detect faces in the frame and determine if they are wearing a face mask or not
			(locations, predictions) = detect_and_predict_mask(normal, face_net, mask_net, args['confidence'])

			# Loop over the detected face locations and their corresponding  locations
			for (box, predic) in zip(locations, predictions):
				# Unpack the bounding box and predictions
				(start_x, start_y, end_x, end_y) = box
				(with_mask, with_mask_no_nose, with_mask_under, without_mask) = predic

				# Determine the class label and color we'll use to draw  the bounding box and text
				max_prob = max(with_mask, with_mask_no_nose, with_mask_under, without_mask)
				if max_prob is with_mask:
					label = 'Mask is OK'
					color = (0, 255, 0)  # Green
				elif max_prob is with_mask_no_nose:
					label = 'Cover your nose'
					color = (15, 219, 250)  # Yellow
				elif max_prob is with_mask_under:
					label = 'Cover yourself'  # Orange
					color = (0, 104, 240)
				else:
					label = 'NO mask'
					color = (0, 0, 255)  # Red

				# Include the probability in the label
				label = '{}: {:.2f}%'.format(label, max_prob * 100)

				# Display the label and bounding box rectangle on the output frame
				cv2.putText(normal, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(normal, (start_x, start_y), (end_x, end_y), color, 2)

			# Decrement the wait counter for every frame where a face is detected
			if len(locations) != 0 and wait_counter != 0:
				wait_counter -= 1

			# Start tracker on the first face detected
			if not tracker_started and len(locations) != 0 and wait_counter == 0:
				tracker_started = True
				bbox = convert_2points_to_1point_and_dims(locations[0])
				track_ok = tracker.init(normal, bbox)

			# Update tracker
			track_ok, bbox = tracker.update(normal)

			# For every frame decrease the counter
			if counter != 0:
				counter -= 1

			# For every set frames, reposition the tracker on the detected face if the distance between their centers are under the threshold value
			if tracker_started and counter == 0 and track_ok:
				counter = args['value']
				tracker_center = get_middle_point(convert_1point_and_dims_to_2points(bbox))
			for box in locations:
				(start_x, start_y, end_x, end_y) = box
				detector_center = get_middle_point(((start_x, start_y), (end_x, end_y)))
				if dist(tracker_center, detector_center) <= args['threshold']:
					bbox = convert_2points_to_1point_and_dims((start_x, start_y, end_x, end_y))
					tracker = create_tracker(args['tracker'])
					track_ok = tracker.init(normal, bbox)
					break

			# Draw bounding box
			if track_ok:
				# Tracking success
				points = convert_1point_and_dims_to_2points(bbox)
				cv2.rectangle(normal, points[0], points[1], (232, 189, 19), 2, 1)
			else:
				# Tracking failure
				cv2.putText(normal, 'Tracking failure detected!', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

			# Display tracker type on frame
			cv2.putText(normal, args['tracker'] + ' Tracker', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

		except NameError:
			pass

		try:
			frame = np.concatenate((normal, thermal), axis=1)
		except ValueError:
			pass
		except NameError:
			pass

		# Display the result
		try:
			cv2.imshow('Video stream', frame)
		except UnboundLocalError:
			pass

		# Exit if Q pressed
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break

	cv2.destroyAllWindows()
	os._exit(1)


def listener():
	rospy.init_node('listener', anonymous=True)
	rospy.Subscriber('/xtion/rgb/image_raw/compressed', CompressedImage, callback_normal)
	rospy.Subscriber('/optris/thermal_image_view/compressed', CompressedImage, callback_thermal)
	video()
	rospy.spin()


if __name__ == '__main__':
	listener()
