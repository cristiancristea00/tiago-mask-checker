from control_msgs.msg import PointHeadGoal
from geometry_msgs.msg import PointStamped
import rospy


def look():
	rospy.init_node('publisher', anonymous = True)
	pub = rospy.Publisher('/head_controller/point_head_action/goal', PointHeadGoal, queue_size = 10)
	point = PointStamped()
	point.point.x = 0
	point.point.y = 0
	point.point.z = 0
	point_head = PointHeadGoal()
	point_head.target = point
	pub.publish(point_head)


if __name__ == '__main__':
	look()
