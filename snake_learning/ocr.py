from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import cv2
from logger import get_logger

EAST = "./frozen_east_text_detection.pb"
LAYER_NAMES = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

MIN_CONFIDENCE = 0.5
PADDING = 0.25
H = 64
W = 64
class OCR:

    @staticmethod
    def decode_predictions(scores, geometry):
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
    
        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the
            # geometrical data used to derive potential bounding box
            # coordinates that surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
    
            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability,
                # ignore it
                if scoresData[x] < MIN_CONFIDENCE:
                    continue
    
                # compute the offset factor as our resulting feature
                # maps will be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
    
                # extract the rotation angle for the prediction and
                # then compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
    
                # use the geometry volume to derive the width and height
                # of the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
    
                # compute both the starting and ending (x, y)-coordinates
                # for the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
    
                # add the bounding box coordinates and probability score
                # to our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
    
        # return a tuple of the bounding boxes and associated confidences
        return (rects, confidences)

    def __init__(self):
        get_logger().info("loading EAST text detector...")
        self.east_net = cv2.dnn.readNet(EAST)

    def _get_boxes(self, image):
        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.east_net.setInput(blob)
        (scores, geometry) = self.east_net.forward(LAYER_NAMES)
        
        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rects, confidences) = OCR.decode_predictions(scores, geometry)
        return non_max_suppression(np.array(rects), probs=confidences)
    

    def get_text(self,image, single_character=False):
        results = self._get_text_in_boxes(image,single_character)
        results = sorted(results, key=lambda r:r[0][1])
        final_text = ""
        for ((_, _, _, _), text) in results:
            final_text += "{} ".format(text)
        return final_text



    def _get_text_in_boxes(self,image, single_character):
        orig = image.copy()
        (h, w) = image.shape[:2]        
        rW = w / float(W)
        rH = h / float(H)
        image = cv2.resize(image, (W, H))
        boxes = self._get_boxes(image)
        results = []

        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            # in order to obtain a better OCR of the text we can potentially
            # apply a bit of padding surrounding the bounding box -- here we
            # are computing the deltas in both the x and y directions
            dX = int((endX - startX) * PADDING)
            dY = int((endY - startY) * PADDING)

            # apply padding to each side of the bounding box, respectively
            startX = max(0, startX - dX)
            startY = max(0, startY - dY)
            endX = min(w, endX + (dX * 2))
            endY = min(h, endY + (dY * 2))

            # extract the actual padded ROI
            roi = orig[startY:endY, startX:endX]
            
            # oem 1 = LSTM only
            # psm 6 = block of text
            params = "-l eng --oem 1 --psm "
            params = params + "10" if single_character else params + "6"
            config = (params)
            text = pytesseract.image_to_string(roi, config=config)

            # add the bounding box coordinates and OCR'd text to the list
            # of results
            results.append(((startX, startY, endX, endY), text))
            
        return results