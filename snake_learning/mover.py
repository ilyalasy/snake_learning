import pyautogui
from logger import get_logger


class Mover:

    def __init__(self, reset_action):
        self.reset_action = reset_action

    def move(self, direction):
        pyautogui.press(direction.name.lower())
        get_logger().info("<{}>".format(direction.name))

    def start_game(self):
        if isinstance(self.reset_action, tuple):
            (x, y) = self.reset_action
            pyautogui.click(x=x, y=y)
        elif isinstance(self.reset_action, str):
            pyautogui.press(self.reset_action.lower())
        else:
            raise TypeError("Cannot perform action of type '{}'".format(
                type(self.reset_action)))
