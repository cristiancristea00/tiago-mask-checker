#!/usr/bin/env python3
from sensor_msgs.msg import CompressedImage
from utils.atomic_wrapper import AtomicWrapper
from utils.face_and_mask_detector import FaceAndMaskDetector
from utils.tracker import Tracker
from utils.temperature_checker import TemperatureChecker
from utils.looker import Looker
from utils.waiting_for_person import WaitingForPerson
from utils.checking_person import CheckingPerson
from utils.geometry import *
from numpy import fromstring, vstack, uint8
from argparse import ArgumentParser
import cv2 as cv
import rospy
import sys

# Defines the command line arguments
arg_parser = ArgumentParser()
arg_parser.add_argument('-t', '--tracker', type = str, default = 'CSRT',
                        help = 'Tracker type: BOOSTING/MIL/KCF/TLD/MEDIANFLOW/MOSSE/CSRT.')
arg_parser.add_argument('-c', '--confidence', type = float, default = 0.5,
                        help = 'Minimum probability to filter weak detections.')
arg_parser.add_argument('-T', '--threshold', type = int, default = 60,
                        help = 'Minimum distance between the bounding boxes of the detector and tracker.')
arg_parser.add_argument('-v', '--value', type = int, default = 1,
                        help = 'Number of frames between tracker and detector sync.')
arg_parser.add_argument('-w', '--wait', type = int, default = 30,
                        help = 'Number of frames to wait before starting the tracker after a face is detected.')
arg_parser.add_argument('-s', '--state', type = int, default = 10,
                        help = 'Number of frames to wait before a message is displayed.')
arg_parser.add_argument('-m', '--move', type = int, default = 4,
                        help = 'Number of frames to wait before the head moves.')
args = vars(arg_parser.parse_args())

# Initialize the atomic wrappers thar are used to acquire the current frame
normal_wrapper = AtomicWrapper()
temp_wrapper = AtomicWrapper()
thermal_wrapper = AtomicWrapper()

# Define the 4 variables that hold the data for the current frame
global image_timestamp
global thermal
global normal
global temp


def callback_normal(data):
    """
    Gets the normal image from the robot, decompress it and crops it to fit
    the temperature data.
    """
    global normal
    global image_timestamp
    normal_wrapper.set(cv.imdecode(fromstring(data.data, uint8), cv.IMREAD_COLOR))
    normal = normal_wrapper.get()[HEIGHT_START:HEIGHT_END, WIDTH_START:WIDTH_END]
    image_timestamp = data.header.stamp


def callback_thermal(data):
    """
    Gets the thermal image from the robot, decompress it and rescale to fit
    the temperature data.
    """
    global thermal
    thermal_wrapper.set(cv.imdecode(fromstring(data.data, uint8), cv.IMREAD_COLOR))
    thermal = cv.resize(thermal_wrapper.get(), (WIDTH_END - WIDTH_START, HEIGHT_END - HEIGHT_START),
                        interpolation = cv.INTER_NEAREST)


def callback_temp(data):
    """
    Gets the temperature data from the robot, decompress it and rescale to
    fit the temperature data.
    """
    global temp
    temp_wrapper.set(cv.imdecode(fromstring(data.data, uint8), cv.IMREAD_ANYDEPTH))
    temp = cv.resize(temp_wrapper.get(), (WIDTH_END - WIDTH_START, HEIGHT_END - HEIGHT_START),
                     interpolation = cv.INTER_NEAREST)


def reset(person_waiter, person_checker, tracker, temp_checker, looker):
    """
    Resets the instances to their initial state.
    """
    person_waiter.reset()
    person_checker.reset()
    temp_checker.reset()
    tracker.reset()
    looker.stop()


def video():
    """
    Principal method of the program that reads the data streams, displays
    the video streams to the user and other messages.
    """
    global image_timestamp
    global thermal
    global normal
    global temp

    # Define the variable that remember the current state: 'waiting' that awaits
    # for a person to enter the frame and 'person_detected' in which checks
    # continuously if the wearer's mask is worn correctly.
    current_state = 'waiting'

    looker = Looker()
    tracker = Tracker(args['tracker'])
    detector = FaceAndMaskDetector(args['confidence'])
    temp_checker = TemperatureChecker()
    person_waiter = WaitingForPerson(tracker, detector, args['wait'])
    person_checker = CheckingPerson(tracker, detector, temp_checker, args['value'], args['wait'], args['threshold'],
                                    args['state'], args['move'])

    while True:
        # Get current frames
        normal_wrapper.set(normal)
        curr_normal = normal_wrapper.get()
        temp_wrapper.set(temp)
        curr_temp = temp_wrapper.get()
        thermal_wrapper.set(thermal)
        curr_thermal = thermal_wrapper.get()

        # While in the 'waiting' state check if a person is in the frame
        if current_state == 'waiting':
            person_waiter.run_prediction(curr_normal)

        # If a person entered the frame, change the current state
        if person_waiter.person_in_frame():
            current_state = 'person_detected'

        # While in the 'person_detected' state check if the person is wearing
        # the mask properly.
        if current_state == 'person_detected':
            person_checker.check_person(curr_normal, curr_temp, looker, image_timestamp)
            if person_checker.mask_ok:
                print(f'{person_checker.temp_checker.get_temp()} C')
                reset(person_waiter, person_checker, tracker, temp_checker, looker)
                looker = Looker()
                current_state = 'waiting'

        frame = vstack((curr_normal, curr_thermal))

        # Display the concatenated current frame
        cv.imshow('Video stream', frame)

        # Exit if Q pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Close the video stream, stops the thread that centers the camera on the
    # face and exits the program
    cv.destroyAllWindows()
    looker.stop()
    sys.exit(0)


def robot():
    """
    Main method that initializes the node and connect to data streams.
    """
    rospy.init_node('robot', anonymous = True)
    rospy.Subscriber('/xtion/rgb/image_raw/compressed', CompressedImage, callback_normal)
    rospy.Subscriber('/optris/thermal_image/compressed', CompressedImage, callback_temp)
    rospy.Subscriber('/optris/thermal_image_view/compressed', CompressedImage, callback_thermal)
    video()


if __name__ == '__main__':
    robot()
