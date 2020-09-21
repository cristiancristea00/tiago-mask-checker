from std_msgs.msg import String
import rospy


class Talker:
    def __init__(self):
        self.talk = rospy.Publisher('/sound_commands', String, queue_size = 3)

    def say(self, sound_file: str):
        self.talk.publish(sound_file)
