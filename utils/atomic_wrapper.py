from threading import Lock


class AtomicWrapper:
	"""
	Wrapper that uses the lock mechanism to acquire the current frame.
	"""

	def __init__(self):
		"""
		Initialize the object and lock.
		"""
		self.obj = None
		self.lock = Lock()

	def set(self, obj):
		"""
		Gets a copy of the object with the lock acquired.
		"""
		with self.lock:
			self.obj = obj.copy()

	def get(self):
		"""
		Returns a copy of the object.
		"""
		return self.obj.copy()
