from typing import Tuple
from control_msgs.msg import PointHeadAction, PointHeadGoal
from geometry_msgs.msg import PointStamped
from rospy import Time
from sensor_msgs.msg import CameraInfo
from utils.geometry import HEIGHT_START, WIDTH_START
from actionlib import SimpleActionClient
from threading import Thread
from numpy import array
import rospy


class Looker(Thread):
    """
    Class that that helps the center the camera on the face.
    """

    def __init__(self):
        """
        Initializes the data stream that sends the instructions to the robot and
        the point that the robot will look at.
        """
        Thread.__init__(self)
        self.daemon = True
        self.running = True

        self.rate = rospy.Rate(10)
        self.point_head_goal = SimpleActionClient('/head_controller/point_head_action', PointHeadAction)

        camera_info_msg = rospy.wait_for_message('/xtion/rgb/camera_info', CameraInfo)
        self.camera_intrinsics = array(camera_info_msg.K).reshape((3, 3))

        self.looker = PointHeadGoal()

        self.looker.target.header.frame_id = '/base_link'
        self.looker.pointing_frame = '/head_2_link'

        self.looker.pointing_axis.x = 1.0
        self.looker.pointing_axis.y = 0.0
        self.looker.pointing_axis.z = 0.0
        self.looker.max_velocity = 0.3

        look_point = PointStamped()
        look_point.header.frame_id = '/base_link'
        look_point.point.x = 25.0
        look_point.point.y = 0.0
        look_point.point.z = 0.0

        self.looker.target = look_point
        self.start()

    @staticmethod
    def correct_point(point: Tuple[int, int]) -> Tuple[int, int]:
        """
        Translate the point according to the cropped image.
        """
        corrected_x = point[0] + WIDTH_START
        corrected_y = point[1] + HEIGHT_START
        return corrected_x, corrected_y

    def run(self):
        """
        Sends continuously the instructions to the robot.
        """
        while self.running:
            self.point_head_goal.send_goal(self.looker)
            self.rate.sleep()

    def stop(self):
        """
        Stops the infinite loop.
        """
        self.running = False
        self.join()

    def point_head(self, point: Tuple[int, int], image_timestamp: Time):
        """
        Sets the goal point to the selected one so the robot will look that way.
        """
        self.running = False

        point = Looker.correct_point(point)

        self.looker.target.header.frame_id = '/xtion_rgb_optical_frame'
        self.looker.pointing_frame = '/xtion_rgb_optical_frame'

        self.looker.pointing_axis.x = 0.0
        self.looker.pointing_axis.y = 0.0
        self.looker.pointing_axis.z = 1.0
        self.looker.max_velocity = 0.2

        look_point = PointStamped()
        look_point.header.frame_id = '/xtion_rgb_optical_frame'
        look_point.point.x = (point[0] - self.camera_intrinsics[0, 2]) / (self.camera_intrinsics[0, 0])
        look_point.point.y = (point[1] - self.camera_intrinsics[1, 2]) / (self.camera_intrinsics[1, 1])
        look_point.point.z = 1.0
        look_point.header.stamp = image_timestamp

        self.looker.target = look_point
        self.point_head_goal.send_goal(self.looker)
