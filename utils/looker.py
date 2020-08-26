from control_msgs.msg import PointHeadActionGoal
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo
from utils.geometry import HEIGHT_START, WIDTH_START
from threading import Thread, Lock
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
        self.set_point_lock = Lock()

        self.rate = rospy.Rate(15)
        self.pub = rospy.Publisher('/head_controller/point_head_action/goal', PointHeadActionGoal, queue_size = 1)

        camera_info_msg = rospy.wait_for_message('/xtion/rgb/camera_info', CameraInfo)
        self.camera_intrinsics = array(camera_info_msg.K).reshape((3, 3))

        self.looker = PointHeadActionGoal()
        self.looker.header.frame_id = '/base_link'
        self.looker.goal.target.header.frame_id = '/base_link'
        self.looker.goal.pointing_frame = '/head_2_link'
        self.looker.goal.pointing_axis.x = 0.0
        self.looker.goal.pointing_axis.y = 0.0
        self.looker.goal.pointing_axis.z = 1.0
        self.looker.goal.max_velocity = 0.3

        look_point = PointStamped()
        look_point.header.frame_id = '/xtion_rgb_optical_frame'
        look_point.point.x = 0.0
        look_point.point.y = 20.5
        look_point.point.z = 1.0

        self.looker.goal.target = look_point

        self.running = True
        self.start()

    @staticmethod
    def correct_point(point):
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
            self.pub.publish(self.looker)
            self.rate.sleep()

    def stop(self):
        """
        Stops the infinite loop.
        """
        self.running = False
        self.join()

    def point_head(self, point):
        """
        Sets the goal point to the selected one so the robot will look that way.
        """
        point = Looker.correct_point(point)
        x = (point[0] - self.camera_intrinsics[0, 2]) / (self.camera_intrinsics[0, 0])
        y = (point[1] - self.camera_intrinsics[1, 2]) / (self.camera_intrinsics[1, 1])
        z = 1.0

        look_point = PointStamped()
        look_point.header.frame_id = '/xtion_rgb_optical_frame'
        look_point.point.x = x * z
        look_point.point.y = y * z
        look_point.point.z = z

        with self.set_point_lock:
            self.looker.goal.target = look_point

    def reset(self):
        """
        Resets the looker to its default parameters.
        """
        look_point = PointStamped()
        look_point.header.frame_id = '/xtion_rgb_optical_frame'
        look_point.point.x = 0.0
        look_point.point.y = 20.5
        look_point.point.z = 1.0

        with self.set_point_lock:
            self.looker.goal.target = look_point
