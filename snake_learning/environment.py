from tensorforce.environments import Environment
from frame_vision import Vision
from mover import Mover
from enums import Action
from ocr import OCR
from logger import get_logger
import time


STATES = {'type': 'float', 'shape': (64, 64, 4)}
ACTIONS = {'type': 'int', 'shape': 1, 'num_actions': 4}


class SnakeEnvironment(Environment):

    """
        Initialize a snake environment.
        Args:
            game_field (dict): Game frame specification dictionary with following attributes (required):
                - top: integer that specifies top position of the game frame
                - left: integer that specifies left position of the game frame
                - width: integer that specifies width of the game frame
                - height: integer that specifies height of the game frame
            game_over_condition (str, list of str, or callable): String or list of strings are treated as words that appears on the screen and indicates the end of the episode. 
                Callable is treated as function to be called to check whether game is finished or not based on screenshot of a current state.
                Should take an image (current screenshot of state) as numpy array and return boolean.
            restart_spec (dict): Dictionary that specifies behavior needed to restart the game with following attributes (required):
                - action: (str, or tuple of int): Action needed to perform to start new episode after death.
                    String will be treated as key name, tuple of ints will be treated as mouse coordinates of a button to be clicked in format (x,y).
                - wait_for (str): String whose appearance on the screen is needed to wait before start of the new episode (default = None).
            preprocess (callable): Function to be called to apply custom preprocess to every screenshot (default=None).
                Should take an image (numpy array) and return preprocessed image.

    """

    def __init__(self, game_field, game_over_condition, restart_spec, preprocess=None):
        
        self.game_over_condition = game_over_condition
        if isinstance(self.game_over_condition, str):
            self.game_over_condition = [self.game_over_condition]

        self.wait_for = None
        if 'wait_for' in restart_spec:
            self.wait_for = restart_spec['wait_for']

        self.logger = get_logger()
        self.start_time = 0.0
        self._started = False
        self._vision = Vision(game_field,preprocess)
        self._ocr = OCR()
        self.mover = Mover(restart_spec['action'])
        self._states = STATES
        self._actions = ACTIONS   

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
        if self.wait_for:
            started = False
            while not started:
                self.mover.start_game()
                time.sleep(0.1)
                image = self._vision.screenshot(grayscale=False,resize=False, normalize=False)
                text = self._ocr.get_text(image)
                started = self.wait_for in text.lower()
            time.sleep(0.3)
        else:
            self.mover.start_game()



    def _get_state(self, is_new_episode):
        return self._vision.get_frames(is_new_episode)

    def close(self):
        pass
    
    def __str__(self):
        return "Snake Environment v.2.0.0"

    def _is_terminal(self):
        image = self._vision.screenshot(grayscale=False,resize=False, normalize=False, preprocess=False)
        
        if callable(self.game_over_condition):
            return self.game_over_condition(image)
        
        text = self._ocr.get_text(image).lower()
        if text:
            self.logger.info("OCR: '{}'".format(text))
        
        terminal = False
        for word in self.game_over_condition:
            terminal = terminal or word in text
        return terminal

    def execute(self, action):   
        action = Action(action[0])

        # old_frame = self._vision.screenshot()
        self.mover.move(action)
        # new_frame = self._vision.screenshot()

        state = self._get_state(False)
        terminal = self._is_terminal()
        reward = self._get_reward(terminal)
        if terminal:
            self._started = False
            self.logger.info("Terminated!")
        return state, terminal, reward
 

    def _get_reward(self, is_terminal):  
        if is_terminal:
            return -600
        lived = time.time() - self.start_time
        return lived
