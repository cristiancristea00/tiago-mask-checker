from utils.geometry import points_to_1point_and_dims


class WaitingForPerson:
	"""
	Class that describes the state in which the robot is waiting for a person
	to enter the frame.
	"""

	def __init__(self, tracker, face_mask, wait_counter_init):
		"""
		Initializes the tracker, detector and the variable that
		stores if a person is in the frame.
		"""
		self.default_wait_counter_init = wait_counter_init
		self.wait_counter = wait_counter_init
		self.detector = face_mask
		self.tracker = tracker
		self.person_detected = False
		self.bounding_box = None

	def run_prediction(self, image):
		"""
		Runs the prediction on the current frame and starts the tracker if
		a person is detected.
		"""
		# Run the prediction on the current frame
		locations, predictions = self.detector.detect_and_predict(image)

		# Decrement the wait counter for every frame where a face is detected
		if len(predictions) != 0 and self.wait_counter != 0:
			self.wait_counter -= 1

		# Start tracker on the first face detected and signal that a face is
		# detected by modifying the variable 'person_detected'
		if not self.person_detected and len(predictions) != 0 and self.wait_counter == 0:
			self.person_detected = True
			self.bounding_box = points_to_1point_and_dims(locations[0])
			self.tracker.track_ok = self.tracker.init(image, self.bounding_box)

	def person_in_frame(self):
		"""
		Checks if a person is in frame.
		"""
		return self.person_detected

	def reset(self):
		"""
		Resets the state to its default parameters.
		"""
		self.wait_counter = self.default_wait_counter_init
		self.person_detected = False
		self.bounding_box = None
