###******************************************************************************************************
# Importing Required Modules and Libraries
###******************************************************************************************************
import cv2
import numpy as np
from openvino.inference_engine import IECore

'''******************************************************************************************************
This code contains the class OpenVinoModel, which initialize parameters and define functions
that allow the estimates of the gaze position for a mouse pointer control
******************************************************************************************************'''

class OpenVinoModel():
    
    '''
    Class that defines parameters and fucntions of an OpenVino pretrained model.
    '''
    def __init__(self):
        '''
        Setting instances variables
        '''
        # Initializing variables
        self.model_weights = None
        self.model_structure = None
        self.device = None
        self.threshold = None
        self.extension = None
        self.markers = None
        self.core = None
        self.model = None

    def load_model(self, model_name, device='CPU', extensions=None, markers=False, threshold=None):
        '''
        Loading the model to the device specified by the user
        '''
        # Assigning values to the model variables
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.extension = extensions
        self.threshold = threshold
        self.markers = markers        

        # Reading the OpenVino model for compatibilitie issues
        try:
            self.core = IECore()
            self.model= self.core.read_network(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=list(self.model.inputs.keys())
        self.input_shape=self.model.inputs[self.input_name[0]].shape
        self.output_name=list(self.model.outputs.keys())
        self.output_shape=self.model.outputs[self.output_name[0]].shape           
        
        # Load the OpenVino model
        self.network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)


    def preprocess_input(self, frame, key=None):
        '''
        Function to adecuate the input images to the OpenVino models of interest
        '''
        pre_frame = cv2.resize(frame, (self.network.inputs[key].shape[3], self.network.inputs[key].shape[2]))                
        pre_frame = pre_frame.transpose((2,0,1))
        pre_frame = pre_frame.reshape(1, *pre_frame.shape)      
    
        return pre_frame


    def predict(self, frame, head_pose_angles=None, key_bias = 0):
        '''
        Function to get predictions from the selected models
        '''
        pre_frames = {}
        for i, feature in enumerate(self.input_name[key_bias:]):
            pre_frames[feature] = self.preprocess_input(frame[i], key=feature)
        outputs = self.network.infer(pre_frames)
        coords, proc_frame = self.preprocess_output(outputs, self.output_name, frame, angles=head_pose_angles)  

        return coords, proc_frame


    def check_model(self):
        raise NotImplementedError

###******************************************************************************************************
#FIN
###******************************************************************************************************
    
