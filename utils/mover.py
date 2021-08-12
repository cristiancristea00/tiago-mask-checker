from geometry_msgs.msg import Twist
from math import pi
import rospy


class Mover:
    """
    Class that incorporates the commands to move the robot.
    """

    def __init__(self):
        """
        Define the publisher where the commands are sent.
        """
        self.base_movement = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size = 15)

    def __move_until_goal_reached(self, speed: float, goal: float, velocity: Twist):
        """
        Internal class that moves the robot until the specified goal is reached.
        """
        time_0 = rospy.Time.now().to_sec()
        current_goal = 0

        while current_goal < goal:
            self.base_movement.publish(velocity)
            time_1 = rospy.Time.now().to_sec()
            current_goal = speed * (time_1 - time_0)

        velocity = Twist()
        self.base_movement.publish(velocity)

    def go_left(self, speed: float, distance: float):
        """
        Makes the robot go to the left by specifying the speed in meters per
        second and the distance in meters.
        """
        self.rotate('counterclockwise', 90, 30)
        self.go('forward', speed, distance)
        self.rotate('clockwise', 90, 30)

    def go_right(self, speed: float, distance: float):
        """
        Makes the robot go to the right by specifying the speed in meters per
        second and the distance in meters.
        """
        self.rotate('clockwise', 90, 30)
        self.go('forward', speed, distance)
        self.rotate('counterclockwise', 90, 30)

    def rotate(self, rotation_type: str, angular_speed: float, angle: float):
        """
        Rotates the robot clockwise/counterclockwise by specifying the angle in
        degrees and the speed in degrees/sec.
        """
        angular_speed = abs(angular_speed)
        angular_speed = angular_speed * 2 * pi / 360
        angle = angle * 2 * pi / 360

        velocity = Twist()

        if rotation_type == 'counterclockwise':
            velocity.angular.z = angular_speed
        elif rotation_type == 'clockwise':
            velocity.angular.z = -angular_speed
        else:
            raise TypeError(F'Unknown rotation type {rotation_type}.')

        self.__move_until_goal_reached(angular_speed, angle, velocity)

    def go(self, movement_type: str, speed: float, distance: float):
        """
        Makes the robot go forward/backward by specifying the speed in meters
        per second and the distance in meters.
        """
        speed = abs(speed)

        velocity = Twist()

        if movement_type == 'forward':
            velocity.linear.x = speed
        elif movement_type == 'backward':
            velocity.linear.x = -speed
        else:
            raise TypeError(F'Unknown movement type {movement_type}.')

        self.__move_until_goal_reached(speed, distance, velocity)
