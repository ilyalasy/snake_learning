import pyautogui
from logger import get_logger

PYTHON_BUTTON = (480,376)
GAME_FIELD = (300,300)

class Mover:

    def __init__(self):
        pass
        
    def move(self, direction):
        pyautogui.press(direction.name.lower())
        get_logger().info("<{}>".format(direction.name))

    def click_center(self):
        pyautogui.click(x=GAME_FIELD[0], y=GAME_FIELD[1])
    
    def start_game(self):
        pyautogui.click(x=PYTHON_BUTTON[0], y=PYTHON_BUTTON[1])
