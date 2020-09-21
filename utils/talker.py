from std_msgs.msg import String
import rospy


class Talker:
    def __init__(self):
        """
        Initializes the Rospy publisher.
        """
        self.talk = rospy.Publisher('/sound_commands', String, queue_size = 3)

    def say(self, sound_file: str):
        """
        Sends the command to the robot to play the specified sound.
        """
        self.talk.publish(sound_file)
