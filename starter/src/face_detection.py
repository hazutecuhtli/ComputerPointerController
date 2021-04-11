###******************************************************************************************************
# Importing Required Modules and Libraries
###******************************************************************************************************
import cv2
import numpy as np
from OpenVinoModel import OpenVinoModel

'''******************************************************************************************************
Class to process images based on the openvino face-detection-adas-0001 model, which has
the following characteristics:

    Inputs

        name: "input" , shape: [1x3x384x672] - An input image in the format [BxCxHxW], where:
            B - batch size
            C - number of channels
            H - image height
            W - image width

        Expected color order is BGR.

    Outputs

        The net outputs blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes.
        For each detection, the description has the format:
            [image_id, label, conf, x_min, y_min, x_max, y_max]
            image_id - ID of the image in the batch
            label - predicted class ID
            conf - confidence for the predicted class
            (x_min, y_min) - coordinates of the top left bounding box corner
            (x_max, y_max) - coordinates of the bottom right bounding box corner
******************************************************************************************************'''


class Model_Face_Detection(OpenVinoModel):
    '''
    Class to detect faces on images based on the face-detection-adas-0001 openvino model.
    '''
    
    def __init__(self):
        '''
        Setting instances variables
        '''        
        OpenVinoModel.__init__(self)

    
    def draw_outputs(self, coords, frame):
        '''
        Function to display intermediate results from this class predictions
        '''
        frame = cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 1)
   

    def preprocess_output(self, outputs, outputs_names, frame, angles=None):
        '''
        Function to process the predicted results using the face-detection-adas-0001 openvino model
        for its use in controlling the mouse cursor
        '''
        coords = []
        frame = frame[0]
        result = outputs[outputs_names[0]]

        for box in result[0][0]:
            conf = box[2]
            if conf >= self.threshold:
                coords.append([box[3], box[4], box[5], box[6]])

        width = frame.shape[1]
        height = frame.shape[0]

        xmin = int(coords[0][0] * width)
        ymin = int(coords[0][1] * height)
        xmax = int(coords[0][2] * width)
        ymax = int(coords[0][3] * height)
    
        return [xmin, ymin, xmax, ymax], frame[ymin:ymax, xmin:xmax]

###******************************************************************************************************
#FIN
###******************************************************************************************************
