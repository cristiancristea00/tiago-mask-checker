from pal_interaction_msgs.msg import TtsAction, TtsGoal
from actionlib import SimpleActionClient


class Talker:
    def __init__(self):
        self.talk = SimpleActionClient('/tts', TtsAction)

    def say(self, text: str):
        goal = TtsGoal()
        goal.rawtext.lang_id = 'en_GB'
        goal.rawtext.text = text
        self.talk.send_goal(goal)
