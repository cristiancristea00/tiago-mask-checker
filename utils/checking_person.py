from typing import List
from numpy import ndarray
from rospy import Time
from utils.face_and_mask_detector import FaceAndMaskDetector
from utils.looker import Looker
from utils.talker import Talker
from utils.temperature_checker import TemperatureChecker
from utils.tracker import Tracker
from utils.waiting_for_person import WaitingForPerson
from utils.geometry import *
import cv2 as cv


class CheckingPerson(WaitingForPerson):
    """
    Class that describes the state in which the robot is checking if the
    person's mask is worn correctly and prompts instructions for the person if
    the mask is worn incorrectly, the checks the temperature.
    """

    def __init__(self, tracker: Tracker, talk: Talker, face_mask: FaceAndMaskDetector, temp_checker: TemperatureChecker,
                 counter_init: int, wait_counter_init: int, dist_threshold: int, state_time: int, move_time: int):
        """
        Besides what 'WaitingForPerson' initializes, it also initializes the
        temperature checker, states dictionary that holds the prediction type
        for some set number of frames, and the variables that hold the
        information if the instruction was said and if the mask is worn
        correctly.
        """
        WaitingForPerson.__init__(self, tracker, face_mask, wait_counter_init)
        self.default_predictions = {'with_mask': 0, 'with_mask_no_nose': 0, 'with_mask_under': 0, 'no_mask': 0}
        self.predictions = self.default_predictions.copy()
        self.default_counter_init = counter_init
        self.default_move_time = move_time
        self.move_time = move_time
        self.counter = counter_init
        self.talker = talk
        self.temp_checker = temp_checker
        self.distance_threshold = dist_threshold
        self.state_time = state_time
        self.action_said = False
        self.mask_ok = False

    @staticmethod
    def prediction_type(prediction: Tuple[float, float, float, float]) -> Tuple[str, float]:
        """
        Gets the prediction type by checking which is the max probability.
        """
        with_mask, with_mask_no_nose, with_mask_under, no_mask = prediction
        probability = max(with_mask, with_mask_no_nose, with_mask_under, no_mask)
        if probability == with_mask:
            return 'with_mask', probability
        elif probability == with_mask_no_nose:
            return 'with_mask_no_nose', probability
        elif probability == with_mask_under:
            return 'with_mask_under', probability
        else:
            return 'no_mask', probability

    @staticmethod
    def draw_detector(locations: List[Tuple[int, int, int, int]], predictions: List[Tuple[float, float, float, float]],
                      image: ndarray):
        """
        Draws the bounding boxes on the detected faces, along with the
        probabilities of the prediction.
        """
        label, color = None, None
        for bounding_box, prediction in zip(locations, predictions):
            # Unpack the bounding box and predictions
            start_x, start_y, end_x, end_y = bounding_box
            prediction_type, probability = CheckingPerson.prediction_type(prediction)
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
            label = f'{label}: {probability * 100:.2f}%'

            # Display the label and bounding box rectangle on the output frame
            cv.putText(image, label, (start_x, start_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv.rectangle(image, (start_x, start_y), (end_x, end_y), color, 2)
            cv.rectangle(image, (start_x, start_y), (end_x, end_y), color, 2)

    @staticmethod
    def draw_tracker(track_ok: bool, image: ndarray, bounding_box: Tuple[int, int, int, int], tracker_type: str):
        """
        Draws the tracker bounding box on the face.
        """
        if track_ok:
            # Tracking success
            points = point_and_dims_to_points(bounding_box)
            cv.rectangle(image, points[0], points[1], (232, 189, 19), 2, 1)
        else:
            # Tracking failure
            cv.putText(image, 'Tracking failure detected!', (5, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display tracker type on frame
        cv.putText(image, tracker_type + ' Tracker', (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50, 170, 50), 2)

    def speak_message(self, prediction_type: str):
        """
        Play the corresponding message based on the prediction type.
        """
        sound_file = None

        if prediction_type == 'with_mask':
            sound_file = 'with_mask'
        elif prediction_type == 'with_mask_no_nose':
            sound_file = 'with_mask_no_nose'
        elif prediction_type == 'with_mask_under':
            sound_file = 'with_mask_under'
        elif prediction_type == 'no_mask':
            sound_file = 'no_mask'

        self.talker.say(sound_file)

    def speak_temperature(self):
        """
        Check if the temperature is below the threshold value and speaks the
        corresponding message.
        """
        if self.temp_checker.get_temp() <= 37.3:
            sound_file = 'temp_okay'
        else:
            sound_file = 'temp_too_high'

        self.talker.say(sound_file)

    def add_prediction(self, prediction_type: str):
        """
        Increment by 1 in the dictionary the prediction type.
        """
        self.predictions[prediction_type] += 1

    def get_max_prediction(self) -> str:
        """
        Gets the maximum prediction.
        """
        return max(self.predictions, key = self.predictions.get)

    def reset_predictions(self):
        """
        Resets the predictions dictionary to its default state.
        """
        self.predictions = self.default_predictions.copy()

    def check_person(self, image: ndarray, temp: ndarray, looker: Looker, image_timestamp: Time):
        """
        Checks the tracked person's mask in the current frame.
        """
        locations, predictions = self.detector.detect_and_predict(image)

        # Draw the detector bounding boxes
        self.draw_detector(locations, predictions, image)

        # Decrement the wait counter for every frame where a face is detected
        if len(predictions) != 0 and self.wait_counter != 0:
            self.wait_counter -= 1

        # Update tracker
        self.tracker.track_ok, self.bounding_box = self.tracker.update(image)

        # For every frame decrease the counter
        if self.counter != 0:
            self.counter -= 1

        # For every set frames, reposition the tracker on the detected face if
        # the distance between their centers are under the threshold value
        if self.counter == 0 and self.tracker.track_ok:
            # Reset the counter to the default value
            self.counter = self.default_counter_init
            # Get the tracker bounding box center
            tracker_center = get_center(point_and_dims_to_points(self.bounding_box))
            for box, prediction in zip(locations, predictions):
                start_x, start_y, end_x, end_y = box
                detector_center = get_center(((start_x, start_y), (end_x, end_y)))
                # Check if the threshold value is met
                if dist(tracker_center, detector_center) <= self.distance_threshold:
                    if self.move_time != 0:
                        self.move_time -= 1
                    self.bounding_box = points_to_point_and_dims((start_x, start_y, end_x, end_y))
                    # Reinitialize the tracker and make the robot look at person
                    # if the set number of frames has passed.
                    self.tracker.create_tracker()
                    self.tracker.track_ok = self.tracker.init(image, self.bounding_box)
                    if self.move_time == 0:
                        looker.point_head(detector_center, image_timestamp)
                        self.move_time = self.default_move_time
                    # Get the temperature for the current frame
                    self.temp_checker.add_data(temp, start_x, start_y, end_x, end_y)
                    # Add the prediction type
                    prediction_type, _ = self.prediction_type(prediction)
                    self.add_prediction(prediction_type)
                    max_state = self.get_max_prediction()
                    # Print the message
                    if not self.action_said and self.predictions[max_state] >= self.state_time:
                        self.speak_message(max_state)
                        self.action_said = True
                        if max_state == 'with_mask':
                            self.mask_ok = True
                    # If the message was already printed, check again if the
                    # mask is worn correctlys
                    elif self.action_said and self.predictions[max_state] >= self.state_time:
                        if max_state == 'with_mask':
                            sound_file = 'with_mask'
                            self.talker.say(sound_file)
                            self.mask_ok = True
                        else:
                            self.reset_predictions()
                    break
        # Draw the tracker bounding box
        self.draw_tracker(self.tracker.track_ok, image, self.bounding_box, self.tracker.name)

    def reset(self):
        """
        Resets the state to its default parameters.
        """
        WaitingForPerson.reset(self)
        self.counter = self.default_counter_init
        self.mask_ok = False
        self.action_said = False
        self.reset_predictions()

    def person_in_frame(self):
        raise AttributeError("'CheckingPerson' has no attribute named 'person_in_frame'")

    def run_prediction(self, image):
        raise AttributeError("'CheckingPerson' has no attribute named 'run_prediction'")
