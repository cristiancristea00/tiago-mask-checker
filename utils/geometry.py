from typing import Tuple
import numpy as np


def points_to_1point_and_dims(bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
	"""Converts opposite points of a rectangle to 1 point and its dimensions,
	width and height."""
	box_x = bbox[0]
	box_y = bbox[1]
	box_width = bbox[2] - bbox[0]
	box_height = bbox[3] - bbox[1]
	return box_x, box_y, box_width, box_height


def point_and_dims_to_2points(bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
	"""Converts 1 point and its dimensions, width and height, of a rectangle to
	opposite points."""
	start_x = int(bbox[0])
	start_y = int(bbox[1])
	end_x = int(bbox[0] + bbox[2])
	end_y = int(bbox[1] + bbox[3])
	return start_x, start_y, end_x, end_y


def get_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
	"""Returns the center point of a rectangle."""
	start_x, start_y, end_x, end_y = bbox
	center_x = int((start_x + end_x) / 2)
	center_y = int((start_y + end_y) / 2)
	return center_x, center_y


def dist(point_1: Tuple[int, int], point_2: Tuple[int, int]) -> int:
	"""Returns the euclidean distance between 2 points."""
	return np.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)
