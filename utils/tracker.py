import cv2 as cv


class Tracker:
	"""Class that acts as a wrapper for the tracker object that makes it easier
	to manipulate it."""

	def __init__(self, name):
		"""Sets the type of the tracker object and creates it."""
		self.name = name
		self.track_ok = False
		self.internal_tracker = None
		self.create_tracker()

	def create_tracker(self):
		"""Sets the tracker type based of its name."""
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

	def init(self, image, bounding_box):
		"""Initializes the tracker on provided bounding box."""
		return self.internal_tracker.init(image, bounding_box)

	def update(self, image):
		"""Updates the tracker on the current bounding box."""
		return self.internal_tracker.update(image)

	def reset(self):
		"""Resets the tracker to its default state."""
		self.create_tracker()
