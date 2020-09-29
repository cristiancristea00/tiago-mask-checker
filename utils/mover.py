#!/usr/bin/env python3
from geometry_msgs.msg import Twist
from math import pi
import rospy


class Mover:
    def __init__(self):
        self.base_movement = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size = 15)

    def go_left(self):
        self.rotate('counterclockwise', 90, 15)
        self.go_forward(1.0, 9)
        self.rotate('clockwise', 90, 15)

    def go_right(self):
        self.rotate('clockwise', 90, 15)
        self.go_forward(1.0, 9)
        self.rotate('counterclockwise', 90, 15)

    def rotate(self, rotation_type: str, angle: int, speed: int):
        angular_speed = speed * 2 * pi / 360
        relative_angle = angle * 2 * pi / 360

        velocity = Twist()

        velocity.linear.x = 0
        velocity.linear.y = 0
        velocity.linear.z = 0
        velocity.angular.x = 0
        velocity.angular.y = 0

        if rotation_type == 'clockwise':
            velocity.angular.z = -abs(angular_speed)
        elif rotation_type == 'counterclockwise':
            velocity.angular.z = abs(angular_speed)
        else:
            raise TypeError(F'Unknown rotation type {rotation_type}.')

        t_0 = rospy.Time.now().to_sec()
        current_angle = 0

        while current_angle < relative_angle:
            self.base_movement.publish(velocity)
            t_1 = rospy.Time.now().to_sec()
            current_angle = angular_speed * (t_1 - t_0)

        velocity = Twist()
        self.base_movement.publish(velocity)

    def go_forward(self, speed: float, time: int):
        velocity = Twist()

        velocity.linear.x = speed
        velocity.linear.y = 0.0
        velocity.linear.z = 0.0

        velocity.angular.x = 0.0
        velocity.angular.y = 0.0
        velocity.angular.z = 0.0

        for _ in range(time):
            self.base_movement.publish(velocity)
            rospy.sleep(0.1)

        velocity = Twist()
        self.base_movement.publish(velocity)


if __name__ == '__main__':
    rospy.init_node('test')
    mover = Mover()
    mover.go_left()
    rospy.sleep(6)
    mover.go_right()
