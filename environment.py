from tensorforce.environments import Environment
from frame_vision import Vision
from mover import Mover
from enums import Action
from ocr import OCR
import time
import random

class SnakeEnvironment(Environment):
    def __init__(self):
        self._started = False
        self._vision = Vision()
        self._ocr = OCR()
        self.mover = Mover()
        self._states = {'type': 'float','shape': (64,64,4)}
        self._actions = {'type': 'int', 'shape': 1, 'num_actions':4}     

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions
        
    def reset(self):
        if not self._started:
            self.mover.start_game()
            time.sleep(2)
            self._started = True
        return self._get_state()

    def _get_state(self):
        return self._vision.get_frames(is_new_episode=True)

    def close(self):
        pass
    
    def __str__(self):
        return "Snake Environment v.2.0.0"

    def _is_terminal(self):
        image = self._vision.screenshot(grayscale=False,resize=False, normalize=False)
        text = self._ocr.get_text(image).lower()
        # VERY BAD!!!
        if text:
            print("####### OCR VALUE ##########")
            print(text)
        return ("game ouer" in text) or ("best score" in text)


    def execute(self, action):   
        action = Action(action[0])

        old_frame = self._vision.screenshot()
        self.mover.move(action)
        new_frame = self._vision.screenshot()

        state = self._vision.get_frames(is_new_episode=False)
        terminal = self._is_terminal()
        reward = self._get_reward(old_frame, new_frame)

        return state, terminal, reward
 

    def _get_reward(self, old_frame, new_frame):
        black = np.array([])
        threshold = 1
        for x in range(old_frame.shape[0]):
            for y in range(old_frame.shape[1])
                pixel = old_frame[x, y]
                if pixel == 255:
                    distance = np.linalg.norm((x,y) - np.mean(black))
                    if distance < threshold:
                        np.append(black, pixel)

       
    


    