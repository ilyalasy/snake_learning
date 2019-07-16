import numpy as np
import cv2
import mss  
from collections import deque


# TODO: Figure out game field automaticaly
# playsnake.org
GAME_FIELD_L = 190
GAME_FIELD_T = 200
GAME_FIELD_W = 380
GAME_FIELD_H = 256

MONITOR = {"top": GAME_FIELD_T, "left": GAME_FIELD_L, "width": GAME_FIELD_W, "height": GAME_FIELD_H}

FRAME_NUMBER = 4
W = 64
H = 64

class Vision:
    def __init__(self):
        self.stacked_frames = deque([np.zeros((W,H), dtype=np.int) for i in range(FRAME_NUMBER)], maxlen=FRAME_NUMBER)

    @staticmethod
    def screenshot(grayscale=True, resize=True, normalize=True):
        mode = cv2.COLOR_BGRA2GRAY if grayscale else cv2.COLOR_BGRA2BGR
        with mss.mss() as sct:
            image = cv2.cvtColor(np.array(sct.grab(MONITOR)), mode)
        if resize:
            image = cv2.resize(image, (W, H))
        if normalize:
            # image = image / 255.0
            image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)[1]
        return image

    def get_frames(self, is_new_episode):     
        frame = Vision.screenshot()

        if is_new_episode:
            # Clear our stacked_frames
            self.stacked_frames = deque([np.zeros((W,H), dtype=np.int) for i in range(FRAME_NUMBER)], maxlen=FRAME_NUMBER)
            
            # Because we're in a new episode, copy the same frame n times
            for i in range(FRAME_NUMBER):
                self.stacked_frames.append(frame) 
        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(frame)
  
        return np.stack(self.stacked_frames, axis=2)
        
        
    

# vision = Vision()
# for i in range(100):
#     try:
#         vision.look()
#     except KeyboardInterrupt:
#         break
# print("FINISHED!")

