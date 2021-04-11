###******************************************************************************************************
# Importing Required Modules and Libraries
###******************************************************************************************************
import cv2
import numpy as np
from OpenVinoModel import OpenVinoModel
from Helper_Functions import RotMat

'''******************************************************************************************************
Class to process images based on the openvino head-pose-estimation-adas-0001 model, which has
the following characteristics:

    Inputs

        name: "data" , shape: [1x3x60x60] - An input image in [1xCxHxW] format. Expected color order is BGR.

    Outputs

    Output layer names in Inference Engine format:

        name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
        name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
        name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).

    Output layer names in Caffe* format:

        name: "fc_y", shape: [1, 1] - Estimated yaw (in degrees).
        name: "fc_p", shape: [1, 1] - Estimated pitch (in degrees).
        name: "fc_r", shape: [1, 1] - Estimated roll (in degrees).

    Each output contains one float value that represents value in Tait-Bryan angles (yaw, pitch or roll).
******************************************************************************************************'''


class Model_HeadPose_Estimation(OpenVinoModel):
    '''
    Class to detect faces head positions based on the head-pose-estimation-adas-0001 openvino model.
    '''
    def __init__(self):
        '''
        Setting instances variables
        '''            
        OpenVinoModel.__init__(self)


    def draw_outputs(self, angles, frame):
        '''
        Function to display intermediate results from this class predictions
        '''    
        vector_length=50
        Origin = [frame.shape[1]/2, frame.shape[0]/2]
        R = RotMat(angles[0][2], angles[0][0], angles[0][1])

        Vectors = [np.array([vector_length, 0, 0]),
                   np.array([0, -vector_length, 0]),
                   np.array([0, 0, -vector_length])]
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        counter = 0
        for Vector, color in zip(Vectors, colors):
            V = np.matmul(R, Vector.transpose())
            if counter == 2:
                large_axisx = -int(V[0])+int(Origin[0])
                large_axisy = -int(V[1])+int(Origin[1])
            else:
                large_axisx = int(Origin[0])
                large_axisy = int(Origin[1])
            cv2.line(frame, (large_axisx, large_axisy),
                    (int(V[0])+int(Origin[0]), int(V[1])+int(Origin[1])), color, 3)            
            counter += 1

            
    def preprocess_output(self, outputs, outputs_names, frame, angles=None):
        '''
        Function to process the predicted results using the head-pose-estimation-adas-0001 openvino model
        for its use in controlling the mouse cursor
        '''
        results = [outputs[feat][0][0] for feat in outputs_names]

        return np.resize(np.array(results), (1,3)), frame[0]

###******************************************************************************************************
#FIN
###******************************************************************************************************
