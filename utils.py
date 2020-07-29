from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from threading import Lock
from math import sqrt
import numpy as np
import cv2 as cv


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

	def detect_and_predict_mask(self, frame):
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
		self.tracker = None
		self.create_tracker()

	def create_tracker(self):
		if self.name == 'BOOSTING':
			self.tracker = cv.TrackerBoosting_create()
		if self.name == 'MIL':
			self.tracker = cv.TrackerMIL_create()
		if self.name == 'KCF':
			self.tracker = cv.TrackerKCF_create()
		if self.name == 'TLD':
			self.tracker = cv.TrackerTLD_create()
		if self.name == 'MEDIANFLOW':
			self.tracker = cv.TrackerMedianFlow_create()
		if self.name == 'MOSSE':
			self.tracker = cv.TrackerMOSSE_create()
		if self.name == 'CSRT':
			self.tracker = cv.TrackerCSRT_create()

	def init(self, image, bbox):
		return self.tracker.init(image, bbox)

	def update(self, image):
		return self.tracker.update(image)


class WaitingForPerson:
	def __init__(self, tracker, face_mask, counter_init, wait_counter_init, distance_threshold):
		self.default_counter_init = counter_init
		self.default_wait_counter_init = wait_counter_init

		self.person_detected = False
		self.track_ok = False

		self.bbox = None

		self.counter = counter_init
		self.wait_counter = wait_counter_init
		self.distance_threshold = distance_threshold

		self.face_mask_detector = face_mask
		self.tracker = tracker

	def run_prediction(self, image, confidence):
		locations, predictions = self.face_mask_detector.detect_and_predict_mask(image)

		# Decrement the wait counter for every frame where a face is detected
		if len(predictions) != 0 and self.wait_counter != 0:
			self.wait_counter -= 1

		# Start tracker on the first face detected
		if not self.person_detected and len(predictions) != 0 and self.wait_counter == 0:
			self.person_detected = True
			self.bbox = convert_2points_to_1point_and_dims(locations[0])
			self.track_ok = self.tracker.init(image, self.bbox)

		# Update tracker
		self.track_ok, self.bbox = self.tracker.update(image)

		# For every frame decrease the counter
		if self.counter != 0:
			self.counter -= 1

		# For every set frames, reposition the tracker on the detected face if the distance between their centers are
		# under the threshold value
		if self.person_detected and self.counter == 0 and self.track_ok:
			self.counter = self.default_counter_init
			tracker_center = get_middle_point(convert_1point_and_dims_to_2points(self.bbox))
			for box in locations:
				(start_x, start_y, end_x, end_y) = box
				detector_center = get_middle_point(((start_x, start_y), (end_x, end_y)))
				if dist(tracker_center, detector_center) <= self.distance_threshold:
					bbox = convert_2points_to_1point_and_dims((start_x, start_y, end_x, end_y))
					self.tracker = self.tracker.create_tracker()
					self.track_ok = self.tracker.init(image, bbox)
					break

	def person_in_frame(self):
		return self.person_detected


class CheckingPerson:
	def __init__(self, tracker, face_mask):
		self.mask_ok = False

		self.tracker = tracker
		self.face_mask_detector = face_mask


# converts opposite points of a rectangle to 1 point and width and height
def convert_2points_to_1point_and_dims(bbox):
	box_x = bbox[0]
	box_y = bbox[1]
	box_width = bbox[2] - bbox[0]
	box_height = bbox[3] - bbox[1]
	return box_x, box_y, box_width, box_height


# the inverse of the above function
def convert_1point_and_dims_to_2points(bbox):
	p1 = (int(bbox[0]), int(bbox[1]))
	p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
	return p1, p2


# returns the middle point of a rectangle
def get_middle_point(bbox):
	(start_x, start_y), (end_x, end_y) = bbox
	middle_x = int((start_x + end_x) / 2)
	middle_y = int((start_y + end_y) / 2)
	return middle_x, middle_y


# return the euclidean distance between 2 points
def dist(p1, p2):
	return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
