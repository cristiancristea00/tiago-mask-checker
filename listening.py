#!/usr/bin/env python
# Rospy node that runs on the robot to listen for commands to play sounds.
from std_msgs.msg import String
import subprocess
import rospy

rospy.init_node('listen_for_sounds', anonymous = True)

base_path = '/home/pal/play_sound/'


def play(data):
    """
    Plays the specified sound.
    """
    subprocess.call(['aplay', base_path + 'sounds/{}.wav'.format(data.data)])


rospy.Subscriber('/sound_commands', String, play)
rospy.spin()
