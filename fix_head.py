from control_msgs.msg import PointHeadActionGoal
from geometry_msgs.msg import Point
import rospy


def look():
	rospy.init_node('publisher', anonymous = True)
	pub = rospy.Publisher('/head_controller/point_head_action/goal', PointHeadActionGoal, queue_size = 1)
	looker = PointHeadActionGoal()
	looker.header.frame_id = '/base_link'
	looker.goal.target.header.frame_id = '/base_link'
	looker.goal.pointing_frame = '/head_2_link'
	looker.goal.max_velocity = 0.3
	look_point = Point()
	look_point.x = 15
	look_point.y = 0
	look_point.z = 0
	looker.goal.target.point = look_point
	while not rospy.is_shutdown():
		pub.publish(looker)


if __name__ == '__main__':
	look()
