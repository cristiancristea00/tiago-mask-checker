from threading import Lock
from numpy import ndarray


class ImageAtomicWrapper:
    """
    Wrapper that uses the lock mechanism to acquire the current frame.
    """

    def __init__(self):
        """
        Initialize the object and lock.
        """
        self.image = None
        self.lock = Lock()

    def set(self, image: ndarray):
        """
        Gets a copy of the object with the lock acquired.
        """
        with self.lock:
            self.image = image.copy()

    def get(self) -> ndarray:
        """
        Returns a copy of the object.
        """
        return self.image.copy()
