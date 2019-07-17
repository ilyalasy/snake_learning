from tensorforce.environments import Environment
from frame_vision import Vision
from mover import Mover
from enums import Action
from ocr import OCR
import time

class SnakeEnvironment(Environment):
    def __init__(self):
        self.start_time = 0.0
        self._started = False
        self._vision = Vision()
        self._ocr = OCR()
        self.mover = Mover()
        self._states = {'type': 'float','shape': (64,64,4)}
        self._actions = {'type': 'int', 'shape': 1, 'num_actions': 4}     

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions
        
    def reset(self):
        if not self._started:   
            self.wait_start()
            
            self.start_time = time.time()
            self._started = True
        return self._get_state(True)

    def wait_start(self):
        self.mover.click_center()
        time.sleep(0.10)
        started = False
        while not started:
            self.mover.start_game()
            time.sleep(0.1)
            image = self._vision.screenshot(grayscale=False,resize=False, normalize=False)
            text = self._ocr.get_text(image)
            started = 'go' in text.lower()



    def _get_state(self, is_new_episode):
        return self._vision.get_frames(is_new_episode)

    def close(self):
        pass
    
    def __str__(self):
        return "Snake Environment v.2.0.0"

    def _is_terminal(self):
        image = self._vision.screenshot(grayscale=False,resize=False, normalize=False)
        text = self._ocr.get_text(image).lower()
        return ("game ouer" in text) or ("best score" in text)


    def execute(self, action):   
        action = Action(action[0])

        old_frame = self._vision.screenshot()
        self.mover.move(action)
        new_frame = self._vision.screenshot()

        state = self._get_state(False)
        terminal = self._is_terminal()
        reward = self._get_reward(terminal)
        if terminal:
            self._started = False
        return state, terminal, reward
 

    def _get_reward(self, is_terminal):  
        if is_terminal:
            return -600
        lived = time.time() - self.start_time
        return lived

       
    


    