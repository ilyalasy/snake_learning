import pyautogui

WORM_BUTTON = (270,360)
GAME_FIELD = (300,300)

class Mover:

    def __init__(self):
        pass
        
    def move(self, direction):
        pyautogui.press(direction.name.lower())

    def click_center(self):
        pyautogui.click(x=GAME_FIELD[0], y=GAME_FIELD[1])
    
    def click_worm(self):
        pyautogui.click(x=WORM_BUTTON[0], y=WORM_BUTTON[1])
