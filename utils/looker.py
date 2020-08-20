from control_msgs.msg import PointHeadActionGoal
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo
from numpy import array
from threading import Thread
import rospy


class Looker(Thread):
	"""
	Class that that helps the center the camera on the face.
	"""

	def __init__(self):
		"""
		Initializes the data stream that sends the instructions to the root and 
		the point that the robot will look at.
		"""
		Thread.__init__(self)
		self.rate = rospy.Rate(5)
		self.pub = rospy.Publisher('/head_controller/point_head_action/goal', PointHeadActionGoal, queue_size = 1)
		self.looker = PointHeadActionGoal()
		self.looker.goal.pointing_axis.x = 0.0
		self.looker.goal.pointing_axis.y = 0.0
		self.looker.goal.pointing_axis.z = 0.0
		self.looker.goal.max_velocity = 0.3

		self.point = PointStamped()
		self.point.header.frame_id = '/xtion_rgb_optical_frame'
		self.point.header.stamp = None

		self.point.point.x = 0.0
		self.point.point.y = 0.0
		self.point.point.z = 15.0

		cam_info_topic = rospy.get_param('/xtion/rgb/camera_info', CameraInfo)
		camera_info_msg = rospy.wait_for_message(cam_info_topic, CameraInfo)
		self.camera_intrinsics = array(camera_info_msg.K).reshape((3, 3))

		self.looker.goal.target.point = self.point

		self.running = True
		self.daemon = True
		self.start()

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

	def point_head(self, point_x, point_y):
		"""
		Sets the goal point to the selected one so the robot will look that way.
		"""
		x = (point_x - self.camera_intrinsics[0, 2]) / (self.camera_intrinsics[0, 0])
		y = (point_y - self.camera_intrinsics[1, 2]) / (self.camera_intrinsics[1, 1])
		z = 1.0

		self.point.point.x = x * z
		self.point.point.y = y * z
		self.point.point.z = z

		self.looker.goal.target.point = self.point

	def reset(self):
		"""
		Resets the looker to its default parameters.
		"""
		self.point.point.x = 0.0
		self.point.point.y = 0.0
		self.point.point.z = 15.0

		self.looker.goal.target.point = self.point
