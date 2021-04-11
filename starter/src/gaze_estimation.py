###******************************************************************************************************
# Importing Required Modules and Libraries
###******************************************************************************************************
import cv2
import numpy as np
from OpenVinoModel import OpenVinoModel

'''******************************************************************************************************
Class to use process images based on the openvino model gaze-estimation-adas-0002, which has
the following characteristics:

    Inputs

        Blob in the format [BxCxHxW] where:
            B - batch size
            C - number of channels
            H - image height
            W - image width

    with the name left_eye_image and the shape [1x3x60x60].

        Blob in the format [BxCxHxW] where:
            B - batch size
            C - number of channels
            H - image height
            W - image width

    with the name right_eye_image and the shape [1x3x60x60].

        Blob in the format [BxC] where:
            B - batch size
            C - number of channels

    with the name head_pose_angles and the shape [1x3].
    Outputs

    The net outputs a blob with the shape: [1, 3], containing Cartesian coordinates of gaze direction vector. Please note that the output vector is not normalizes and has non-unit length.

    Output layer name in Inference Engine format:

    gaze_vector

    Output layer name in Caffe2 format:

    gaze_vector
******************************************************************************************************'''

class Model_Gaze_Estimation(OpenVinoModel):
    '''
    Class to estimate the gaze direction based on the gaze-estimation-adas-0002 openvino model.
    '''
    def __init__(self):
        '''
        Setting instances variables
        '''         
        OpenVinoModel.__init__(self)
   

    def preprocess_output(self, outputs, outputs_names, frame, angles=None):
        '''
        Function to process the predicted results using the gaze-estimation-adas-0002 openvino model
        for its use on controlling the mouse cursor
        '''
        gaze_vector = outputs[outputs_names[0]][0]
        angles = angles[0]
        
        roll = angles[1] * np.pi/180.0
        pitch = angles[0] * np.pi/180.0        
        yaw = angles[2] * np.pi/180.0

        mouse_x = gaze_vector[0] * np.cos(roll) + gaze_vector[1] * np.sin(roll)
        mouse_y = -gaze_vector[0] * np.sin(roll) + gaze_vector[1] * np.cos(roll)        

        return [gaze_vector, (mouse_x, mouse_y)], frame

###******************************************************************************************************
#FIN
###******************************************************************************************************
