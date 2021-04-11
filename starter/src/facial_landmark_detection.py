###******************************************************************************************************
# Importing Required Modules and Libraries
###******************************************************************************************************
import cv2
import numpy as np
from OpenVinoModel import OpenVinoModel

'''******************************************************************************************************
Class to process images based on the openvino landmarks-regression-retail-0009 model, which has
the following characteristics:

    Inputs

    Name: "data" , shape: [1x3x48x48] - An input image in the format [BxCxHxW], where:

        B - batch size
        C - number of channels
        H - image height
        W - image width

    The expected color order is BGR.
    
    Outputs

        The net outputs a blob with the shape: [1, 10], containing a row-vector of 10 floating
        point values for five landmarks coordinates in the form (x0, y0, x1, y1, ..., x4, y4). All
        the coordinates are normalized to be in range [0,1].
******************************************************************************************************'''

class Model_FacialLandMark_Detection(OpenVinoModel):
    '''
    Class to detect faces on images based on the landmarks-regression-retail-0009 openvino model.
    '''
    def __init__(self):
        '''
        Setting instances variables
        '''         
        OpenVinoModel.__init__(self)


    def draw_outputs(self, eye_coords, frame, boxes_face):
        '''
        Function to display intermediate results from this class predictions
        '''
        width = boxes_face[0]
        height = boxes_face[1] 
        
        frame_left = cv2.rectangle(frame, (eye_coords[0][0] + width, eye_coords[0][1] + height),
                                   (eye_coords[0][2] + width, eye_coords[0][3] + height), (0, 255, 0), 1)
        frame_right = cv2.rectangle(frame, (eye_coords[1][0] + width, eye_coords[1][1] + height),
                                   (eye_coords[1][2] + width, eye_coords[1][3] + height), (0, 255, 0), 1)        


    def preprocess_output(self, outputs, outputs_names, frame, angles=None):
        '''
        Function to process the predicted results using the landmarks-regression-retail-0009 openvino model
        for its use in controlling the mouse cursor
        '''
        box_dim = 20
        frame = frame[0]    
        w = frame.shape[1]
        h = frame.shape[0]    

        coords = [outputs[feat][0] for feat in outputs_names]
        coords = np.array([coords[0][0][0][0], coords[0][1][0][0], coords[0][2][0][0], coords[0][3][0][0]])
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32)

        le_xmin = coords[0] - box_dim
        le_ymin = coords[1] - box_dim
        le_xmax = coords[0] + box_dim
        le_ymax = coords[1] + box_dim
        
        re_xmin = coords[2] - box_dim
        re_ymin = coords[3] - box_dim
        re_xmax = coords[2] + box_dim
        re_ymax = coords[3] + box_dim

        le = frame[le_ymin:le_ymax, le_xmin:le_xmax]
        re = frame[re_ymin:re_ymax, re_xmin:re_xmax]
        
        eye_coords = [[le_xmin, le_ymin, le_xmax, le_ymax], [re_xmin, re_ymin, re_xmax, re_ymax]]

        return eye_coords, [le, re]

###******************************************************************************************************
#FIN
###******************************************************************************************************
