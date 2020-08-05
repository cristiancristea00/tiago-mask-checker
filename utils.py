from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from control_msgs.msg import PointHeadActionGoal
from geometry_msgs.msg import Point
from threading import Lock, Thread
from math import sqrt
import numpy as np
import cv2 as cv
import rospy


class AtomicWrapper:
	def __init__(self):
		self.obj = None
		self.lock = Lock()

	def set(self, obj):
		with self.lock:
			self.obj = obj.copy()

	def get(self):
		return self.obj.copy()


class FaceAndMaskDetector:
	def __init__(self, confidence):
		self.face_net = cv.dnn.readNet('FaceNet.prototxt', 'FaceNet.caffemodel')
		self.mask_net = load_model('mask_detector.model')
		self.confidence = confidence

	def detect_and_predict(self, frame):
		# grab the dimensions of the frame and then construct a blob
		# from it
		(h, w) = frame.shape[:2]
		blob = cv.dnn.blobFromImage(frame, 1.0, (235, 350))

		# pass the blob through the network and obtain the face detections
		self.face_net.setInput(blob)
		detections = self.face_net.forward()

		# initialize our list of faces, their corresponding locations,
		# and the list of predictions from our face mask network
		faces = []
		locations = []
		predictions = []

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the detection
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the confidence is
			# greater than the minimum confidence
			if confidence > self.confidence:
				# compute the (x, y)-coordinates of the bounding box for
				# the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(start_x, start_y, end_x, end_y) = box.astype('int')

				# ensure the bounding boxes fall within the dimensions of
				# the frame
				(start_x, start_y) = (max(0, start_x), max(0, start_y))
				(end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

				# extract the face ROI, convert it from BGR to RGB channel
				# ordering, resize it to 224x224, and pre-process it
				face = frame[start_y:end_y, start_x:end_x]
				face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
				face = cv.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)
				face = np.expand_dims(face, axis = 0)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locations.append((start_x, start_y, end_x, end_y))

		# only make a predictions if at least one face was detected
		if len(faces) > 0:
			# for faster inference we'll make batch predictions on *all*
			# faces at the same time rather than one-by-one predictions
			# in the above `for` loop
			predictions = self.mask_net.predict(faces)

		# return a 2-tuple of the face locations and their corresponding
		# locations
		return locations, predictions


class Tracker:
	def __init__(self, name):
		self.name = name
		self.track_ok = False
		self.internal_tracker = None
		self.create_tracker()

	def create_tracker(self):
		if self.name == 'CSRT':
			self.internal_tracker = cv.TrackerCSRT_create()
		elif self.name == 'BOOSTING':
			self.internal_tracker = cv.TrackerBoosting_create()
		elif self.name == 'MIL':
			self.internal_tracker = cv.TrackerMIL_create()
		elif self.name == 'KCF':
			self.internal_tracker = cv.TrackerKCF_create()
		elif self.name == 'TLD':
			self.internal_tracker = cv.TrackerTLD_create()
		elif self.name == 'MEDIANFLOW':
			self.internal_tracker = cv.TrackerMedianFlow_create()
		elif self.name == 'MOSSE':
			self.internal_tracker = cv.TrackerMOSSE_create()

	def init(self, image, bbox):
		return self.internal_tracker.init(image, bbox)

	def update(self, image):
		return self.internal_tracker.update(image)

	def reset(self):
		self.create_tracker()


class TemperatureChecker:
	def __init__(self):
		self.temp_data = None
		self.temps = []

	def add_data(self, image, start_x, start_y, end_x, end_y):
		self.temp_data = image
		try:
			t = np.max(image[start_y + 15:end_y - 35, start_x + 14:end_x - 14] - 1000) / 10.0
		except ValueError:
			t = 0
		if t > 34:
			self.temps.append(t)

	def get_temp(self):
		return np.round(np.mean(self.temps), 1)

	def reset(self):
		self.temps.clear()


class WaitingForPerson:
	def __init__(self, tracker, face_mask, wait_counter_init):
		self.default_wait_counter_init = wait_counter_init
		self.person_detected = False
		self.bbox = None
		self.wait_counter = wait_counter_init
		self.detector = face_mask
		self.tracker = tracker

	def run_prediction(self, image):
		locations, predictions = self.detector.detect_and_predict(image)

		# Decrement the wait counter for every frame where a face is detected
		if len(predictions) != 0 and self.wait_counter != 0:
			self.wait_counter -= 1

		# Start tracker on the first face detected
		if not self.person_detected and len(predictions) != 0 and self.wait_counter == 0:
			self.person_detected = True
			self.bbox = points_to_1point_and_dims(locations[0])
			self.tracker.track_ok = self.tracker.init(image, self.bbox)

	def person_in_frame(self):
		return self.person_detected

	def reset(self):
		self.wait_counter = self.default_wait_counter_init

		self.person_detected = False
		self.bbox = None


class CheckingPerson(WaitingForPerson):
	def __init__(self, tracker, face_mask, temp_checker, counter_init, wait_counter_init, dist_threshold, state_time):
		WaitingForPerson.__init__(self, tracker, face_mask, wait_counter_init)
		self.mask_ok = False
		self.action_said = False
		self.default_counter_init = counter_init
		self.default_states = {'with_mask': 0, 'with_mask_no_nose': 0, 'with_mask_under': 0, 'no_mask': 0}
		self.states = self.default_states.copy()
		self.state_time = state_time
		self.distance_threshold = dist_threshold
		self.counter = counter_init
		self.temp_checker = temp_checker

	def add_state(self, state):
		self.states[state] += 1

	def get_max_state(self):
		return max(self.states, key = self.states.get)

	def reset_states(self):
		self.states = self.default_states.copy()

	@staticmethod
	def prediction_type(prediction):
		with_mask, with_mask_no_nose, with_mask_under, no_mask = prediction
		max_prob = max(with_mask, with_mask_no_nose, with_mask_under, no_mask)
		if max_prob == with_mask:
			return 'with_mask', max_prob
		elif max_prob == with_mask_no_nose:
			return 'with_mask_no_nose', max_prob
		elif max_prob == with_mask_under:
			return 'with_mask_under', max_prob
		else:
			return 'no_mask', max_prob

	@staticmethod
	def print_message(state):
		if state == 'with_mask':
			print('Your mask is OK. Let\'s check your temperature now.')
		elif state == 'with_mask_no_nose':
			print('Please cover your nose.')
		elif state == 'with_mask_under':
			print('Please don\'t user your mask as a chin guard.')
		elif state == 'no_mask':
			print('You can\'t enter without a mask.')

	@staticmethod
	def draw_detector(locations, predictions, image):
		label, color = None, None
		for bbox, prediction in zip(locations, predictions):
			# Unpack the bounding box and predictions
			start_x, start_y, end_x, end_y = bbox
			prediction_type, max_prob = CheckingPerson.prediction_type(prediction)
			if prediction_type == 'with_mask':
				label = 'Mask is OK'
				color = (0, 255, 0)  # Green
			elif prediction_type == 'with_mask_no_nose':
				label = 'Cover your nose'
				color = (15, 219, 250)  # Yellow
			elif prediction_type == 'with_mask_under':
				label = 'Cover yourself'  # Orange
				color = (0, 104, 240)
			elif prediction_type == 'no_mask':
				label = 'NO mask'
				color = (0, 0, 255)  # Red

			# Include the probability in the label
			label = f'{label}: {max_prob * 100:.2f}%'

			# Display the label and bounding box rectangle on the output frame
			cv.putText(image, label, (start_x, start_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv.rectangle(image, (start_x, start_y), (end_x, end_y), color, 2)
			cv.rectangle(image, (start_x, start_y), (end_x, end_y), color, 2)

	@staticmethod
	def draw_tracker(track_ok, image, bbox, tracker_type):
		if track_ok:
			# Tracking success
			points = point_and_dims_to_2points(bbox)
			cv.rectangle(image, points[0], points[1], (232, 189, 19), 2, 1)
		else:
			# Tracking failure
			cv.putText(image, 'Tracking failure detected!', (5, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

		# Display tracker type on frame
		cv.putText(image, tracker_type + ' Tracker', (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50, 170, 50), 2)

	def check_person(self, image, temp):
		locations, predictions = self.detector.detect_and_predict(image)

		self.draw_detector(locations, predictions, image)

		# Decrement the wait counter for every frame where a face is detected
		if len(predictions) != 0 and self.wait_counter != 0:
			self.wait_counter -= 1

		# Update tracker
		self.tracker.track_ok, self.bbox = self.tracker.update(image)

		# For every frame decrease the counter
		if self.counter != 0:
			self.counter -= 1

		# For every set frames, reposition the tracker on the detected face if
		# the distance between their centers are under the threshold value
		if self.counter == 0 and self.tracker.track_ok:
			self.counter = self.default_counter_init
			tracker_center = get_center(point_and_dims_to_2points(self.bbox))
			for box, prediction in zip(locations, predictions):
				start_x, start_y, end_x, end_y = box
				detector_center = get_center(((start_x, start_y), (end_x, end_y)))
				if dist(tracker_center, detector_center) <= self.distance_threshold:
					self.bbox = points_to_1point_and_dims((start_x, start_y, end_x, end_y))
					self.tracker.create_tracker()
					self.tracker.track_ok = self.tracker.init(image, self.bbox)

					self.temp_checker.add_data(temp, start_x, start_y, end_x, end_y)
					prediction_type, _ = self.prediction_type(prediction)
					self.add_state(prediction_type)
					max_state = self.get_max_state()
					if not self.action_said and self.states[max_state] >= self.state_time:
						self.print_message(max_state)
						self.action_said = True
						if max_state == 'with_mask':
							self.mask_ok = True
					elif self.action_said and self.states[max_state] >= self.state_time:
						if max_state == 'with_mask':
							print('You are okay now. Let\'s check your temperature.')
							self.mask_ok = True
						else:
							self.reset_states()
					break

		self.draw_tracker(self.tracker.track_ok, image, self.bbox, self.tracker.name)

	def reset(self):
		WaitingForPerson.reset(self)
		self.counter = self.default_counter_init
		self.mask_ok = False
		self.action_said = False
		self.states = self.default_states.copy()

	def person_in_frame(self):
		raise AttributeError('\'CheckingPerson\' has no attribute named \'person_in_frame\'')

	def run_prediction(self, image):
		raise AttributeError('\'CheckingPerson\' has no attribute named \'run_prediction\'')


# converts opposite points of a rectangle to 1 point and width and height
def points_to_1point_and_dims(bbox):
	box_x = bbox[0]
	box_y = bbox[1]
	box_width = bbox[2] - bbox[0]
	box_height = bbox[3] - bbox[1]
	return box_x, box_y, box_width, box_height


# the inverse of the above function
def point_and_dims_to_2points(bbox):
	p1 = (int(bbox[0]), int(bbox[1]))
	p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
	return p1, p2


# returns the middle point of a rectangle
def get_center(bbox):
	(start_x, start_y), (end_x, end_y) = bbox
	center_x = int((start_x + end_x) / 2)
	center_y = int((start_y + end_y) / 2)
	return center_x, center_y


# return the euclidean distance between 2 points
def dist(p1, p2):
	return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def reset(person_waiter, person_checker, tracker, temp_checker):
	person_waiter.reset()
	person_checker.reset()
	tracker.reset()
	temp_checker.reset()


class Look:
	def __init__(self):
		self.pub = rospy.Publisher('/head_controller/point_head_action/goal', PointHeadActionGoal, queue_size = 1)
		self.looker = PointHeadActionGoal()
		self.looker.header.frame_id = '/base_link'
		self.looker.goal.target.header.frame_id = '/base_link'
		self.looker.goal.pointing_frame = '/head_2_link'
		self.looker.goal.max_velocity = 0.3
		self.look_point = Point()
		self.look_point.x = 15
		self.look_point.y = 0
		self.look_point.z = 0
		self.looker.goal.target.point = self.look_point
		self.r = rospy.Rate(5)
		self.running = True
		self.look = Thread(target = self.run(), daemon = True)

	def run(self):
		while self.running:
			self.pub.publish(self.looker)
			self.r.sleep()

	def start(self):
		self.look.start()

	def stop(self):
		self.look.join()
