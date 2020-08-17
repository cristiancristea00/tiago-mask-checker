from control_msgs.msg import PointHeadActionGoal
from geometry_msgs.msg import Point
from threading import Thread
import rospy


class Looker(Thread):
	"""Class that that helps the center the camera on the face."""

	def __init__(self):
		"""Initializes the data stream that sends the instructions to the root
		and the point that the robot will look at."""
		Thread.__init__(self)
		self.pub = rospy.Publisher('/head_controller/point_head_action/goal', PointHeadActionGoal, queue_size = 1)
		self.looker = PointHeadActionGoal()
		self.looker.header.frame_id = '/base_link'
		self.looker.goal.target.header.frame_id = '/base_link'
		self.looker.goal.pointing_frame = '/head_2_link'
		self.looker.goal.max_velocity = 0.2
		self.look_point = Point()
		self.look_point.x = 15
		self.look_point.y = 0
		self.look_point.z = 0
		self.looker.goal.target.point = self.look_point
		self.rate = rospy.Rate(5)
		self.running = True
		self.daemon = True
		self.start()

	def run(self):
		"""Sends continuously the instructions to the robot."""
		while self.running:
			self.pub.publish(self.looker)
			self.rate.sleep()

	def stop(self):
		"""Stops the infinite loop."""
		self.running = False
