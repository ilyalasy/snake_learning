import numpy as np
import imutils
import pyautogui
import cv2
import mss  
from enums import Action, Direction

# template = cv2.imread("images/snake_part.png")
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# template = cv2.Canny(template, 50, 200)
# cv2.imwrite("images/snake_part.png", template)
# image = cv2.imread("images/main.png")



# TODO: Figure out game field automaticaly
# playsnake.org
GAME_FIELD_L = 120
GAME_FIELD_T = 200
GAME_FIELD_W = 380
GAME_FIELD_H = 240

MONITOR = {"top": GAME_FIELD_T, "left": GAME_FIELD_L, "width": GAME_FIELD_W, "height": GAME_FIELD_H}

# with mss.mss() as sct:
#     image = cv2.cvtColor(np.array(sct.grab(MONITOR)), cv2.COLOR_RGB2BGR)
#     cv2.imshow("Image",image)
#     cv2.waitKey(0)

APPLE = "images/apple.png"
SNAKE = "images/snake_part.png"

class Vision:
    def __init__(self):
        self.current_direction = Direction.DOWN
        self.head_pos = None
        self.apple_pos = None
        self.current_shot = {}
        self.binary = {}

    def get_new_direction(self, action):
        if action is None or action == Action.FORWARD:
            return self.current_direction
        if action == Action.RIGHT:
            return Direction((self.current_direction.value + 1) % 4 )
        if action == Action.LEFT:

            return Direction(3 if (self.current_direction.value - 1) <= 0 else self.current_direction.value - 1)     

    def look(self, action = None):
        self.current_shot = self.take_screenshot()
        self.binary = cv2.threshold(self.current_shot.copy(), 60, 255, cv2.THRESH_BINARY)[1]             
        self.apple_pos = self.get_apple(action)
        self.head_pos = self.get_head(action)
        self.current_direction = self.get_new_direction(action)

    def take_screenshot(self):

        with mss.mss() as sct:
            image = cv2.cvtColor(np.array(sct.grab(MONITOR)), cv2.COLOR_RGB2BGR)
        
        # cv2.imwrite("images/test/-1.png", image)

        # image = cv2.imread("images/main.png")        
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def get_template(self, template_path):
        template = cv2.imread(template_path)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        (tH, tW) = template.shape[:2]
        found = None
        # loop over the scales of the image
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(self.current_shot, width = int(self.current_shot.shape[1] * scale))
            r = self.current_shot.shape[1] / float(resized.shape[1])

            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # if we have found a new maximum correlation value, then update
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)

        # unpack the bookkeeping variable and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

        center = {'x' : int(startX + tW / 2), 'y' : int(startY + tH / 2)}
        return (center, tW, tH)
        # # draw a bounding box around the detected result and display the image
        # cv2.rectangle(image, (startX, startY), (endX, endY), (0, (i) * 255, 255), 2)

    def get_apple(self, action):
        if action is None or self.is_apple(action):
            return self.get_template(APPLE)
        return self.apple_pos

    def get_head(self, action):
        if action is None:
            return self.get_template(SNAKE)
        return self.get_next_center(action)


    def get_next_center(self, action):
        (center, w, h) = self.head_pos
        next_direction = self.get_new_direction(action)
        new_center = center.copy()
        if next_direction == Direction.DOWN:
            new_center['y'] = new_center['y'] + h
        if next_direction == Direction.UP:
            new_center['y'] = new_center['y'] - h
        if next_direction == Direction.LEFT:
            new_center['x'] = new_center['x'] - w
        if next_direction == Direction.RIGHT:
            new_center['x'] = new_center['x'] + w
        return (new_center, w, h)

    def get_next_area(self, action):
        (next_center,w,h) = self.get_next_center(action)
        return {'x1': int(next_center['x'] - w/2), 'x2': int(next_center['x'] + w/2), 
                'y1': int(next_center['y'] - h/2), 'y2': int(next_center['y'] + h/2)}
    
    def is_clear(self, action):
        if self.is_apple(action):
            return True
        area = self.get_next_area(action)
        return 0 not in self.binary[area['x1'] : area['x2'], area['y1'] : area['y2']]

    def is_apple(self,action):
        area = self.get_next_area(action)
        return self.apple_pos[0]['x'] in range(area['x1'], area['x2']) and self.apple_pos[0]['y'] in range(area['y1'], area['y2'])

     
