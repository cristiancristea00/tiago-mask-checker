from control_msgs.msg import PointHeadActionGoal
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo
from threading import Thread
from numpy import array
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
		self.pub = rospy.Publisher('/head_controller/point_head_action/goal', PointHeadActionGoal, queue_size = 15)
		self.looker = PointHeadActionGoal()
		self.looker.goal.pointing_frame = '/head_2_link'
		self.looker.header.frame_id = '/base_link'
		self.looker.goal.target.header.frame_id = '/base_link'
		self.looker.goal.pointing_axis.x = 0.0
		self.looker.goal.pointing_axis.y = 0.0
		self.looker.goal.pointing_axis.z = 1.0
		self.looker.goal.max_velocity = 0.3

		self.look_point = PointStamped()
		self.look_point.header.frame_id = '/xtion_rgb_optical_frame'
		self.look_point.point.x = 13
		self.look_point.point.y = 0.0
		self.look_point.point.z = 0.0

		camera_info_msg = rospy.wait_for_message('/xtion/rgb/camera_info', CameraInfo)
		self.camera_intrinsics = array(camera_info_msg.K).reshape((3, 3))

		self.looker.goal.target = self.look_point

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

	def point_head(self, point):
		"""
		Sets the goal point to the selected one so the robot will look that way.
		"""
		x = (point[0] - self.camera_intrinsics[0, 2]) / (self.camera_intrinsics[0, 0])
		y = (point[1] - self.camera_intrinsics[1, 2]) / (self.camera_intrinsics[1, 1])
		z = 1.0

		self.look_point.point.x = x * z
		self.look_point.point.y = y * z
		self.look_point.point.z = z

		self.looker.goal.target = self.look_point

	def reset(self):
		"""
		Resets the looker to its default parameters.
		"""
		self.look_point.point.x = 0.0
		self.look_point.point.y = 0.0
		self.look_point.point.z = 0.0

		self.looker.goal.target.look_point = self.look_point
