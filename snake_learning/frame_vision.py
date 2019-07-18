import numpy as np
import cv2
import mss
from collections import deque


FRAME_NUMBER = 4
W = 64
H = 64


class Vision:
    def __init__(self, monitor):
        self.monitor = monitor
        self.stacked_frames = deque(
            [np.zeros((W, H), dtype=np.int) for i in range(FRAME_NUMBER)], maxlen=FRAME_NUMBER)

    def screenshot(self, grayscale=True, resize=True, normalize=True):
        mode = cv2.COLOR_BGRA2GRAY if grayscale else cv2.COLOR_BGRA2BGR
        with mss.mss() as sct:
            image = cv2.cvtColor(np.array(sct.grab(self.monitor)), mode)
        if resize:
            image = cv2.resize(image, (W, H))
        if normalize:
            image = image / 255.0
        
        cv2.imwrite("./lol.png",image)
        return image

    def get_frames(self, is_new_episode):
        frame = self.screenshot()

        if is_new_episode:
            # Clear our stacked_frames
            self.stacked_frames = deque(
                [np.zeros((W, H), dtype=np.int) for i in range(FRAME_NUMBER)], maxlen=FRAME_NUMBER)

            # Because we're in a new episode, copy the same frame n times
            for i in range(FRAME_NUMBER):
                self.stacked_frames.append(frame)
        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(frame)

        return np.stack(self.stacked_frames, axis=2)
