from numpy import ndarray
import numpy as np
import warnings


class TemperatureChecker:
    """
    Class that stores the temperature data for the current person.
    """

    def __init__(self):
        self.temp_data = []

    def add_data(self, image: ndarray, start_x: int, start_y: int, end_x: int, end_y: int):
        """
        Adds the temperature data from the forehead for the current frame.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category = RuntimeWarning)
            t = np.mean(image[start_y + 15:end_y - 30, start_x + 14:end_x - 14] - 1000) / 10.0
            if t > 34:
                self.temp_data.append(t)

    def get_temp(self) -> float:
        """
        Gets the temperature as a mean of the acquired data while the person
        was in the frame.
        """
        return np.round(np.mean(self.temp_data), 1)

    def reset(self):
        """
        Resets the temperature checker to its default state.
        """
        self.temp_data.clear()
