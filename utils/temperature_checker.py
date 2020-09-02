import numpy as np


class TemperatureChecker:
    """
    Class that stores the temperature data for the current person.
    """

    def __init__(self):
        self.temp_data = []

    def add_data(self, image, start_x, start_y, end_x, end_y):
        """
        Adds the temperature data from the forehead for the current frame.
        """
        try:
            t = np.max(image[start_y + 23:end_y - 43, start_x + 15:end_x - 15] - 1000) / 10.0
        except ValueError:
            t = 0
        if t > 34:
            self.temp_data.append(t)

    def get_temp(self):
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
