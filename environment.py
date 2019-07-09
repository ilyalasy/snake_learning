from tensorforce.environments import Environment
from template_vision import Vision
from mover import Mover
from enums import Action
import time

class SnakeEnvironment(Environment):
    def __init__(self):
        self._started = False
        self._vision = Vision()
        self.mover = Mover()
        self._states = {'type': 'float','shape': 6}
        self._actions = {'type': 'int', 'shape': 1, 'num_actions':3}     

    @property
    def states(self):
        return self._states

    @property
    def actions(self):
        return self._actions
        
    def reset(self):
        if not self._started:
            self.mover.click_worm()
            time.sleep(2)
            self._vision.look()
            self._started = True
        return self._get_state()

    def _get_state(self):
        return [float(self._vision.is_clear(Action.FORWARD)), float(self._vision.is_clear(Action.LEFT)), float(self._vision.is_clear(Action.RIGHT)),
                float(self._vision.is_apple(Action.FORWARD)), float(self._vision.is_apple(Action.LEFT)), float(self._vision.is_apple(Action.RIGHT))]

    def close(self):
        pass
    
    def __str__(self):
        return "Snake Environment v.1.0.0"

    def execute(self, action):   
        action = Action(action[0])
        terminal = not self._vision.is_clear(action)

        direction = self._vision.get_new_direction(action)
        self.mover.move(direction)
        (old_center, _, _) = self._vision.head_pos
        (apple_center,_,_) = self._vision.apple_pos        
        self._vision.look(action)
        (new_center, _,_) = self._vision.head_pos

        state = self._get_state()
        reward = self._get_reward(old_center, new_center, apple_center)
        return state, terminal, reward


    def _get_reward(self, old_center, new_center, apple_center):
        old_diff = {key: apple_center[key] - old_center.get(key, 0) for key in apple_center.keys()}
        new_diff = {key: apple_center[key] - new_center.get(key, 0) for key in apple_center.keys()}

        if new_diff.values() == [0, 0]:
            return 2.0
        if new_diff['x'] > old_diff['x'] or new_diff['y'] > old_diff['y']:
            return -1.5
        if new_diff['x'] < old_diff['x'] or new_diff['y'] < old_diff['y']:
            return 1.5

    


    