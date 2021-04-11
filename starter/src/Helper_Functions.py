###******************************************************************************************************
# Importing Required Modules and Libraries
###******************************************************************************************************
import numpy as np
'''******************************************************************************************************
This file contains functions needed for the correct implementation of the Edge AI udacity final project
******************************************************************************************************'''

def RotMat(gamma, beta, alpha):
    '''
    Function that calculate a rotation matrix based on the roll, pitch an yaw angles

    inputs:

        gamma --> angle that represents the roll angle
        beta  --> angle that represents the pitch angle
        alpha --> angle that represents the yaw angle

    output

        R --> Calculated rotation matrix
    '''

    # Converting the rotation angles to radians
    gamma = gamma * np.pi/180.0
    beta = beta * np.pi/180.0
    alpha = alpha * np.pi/180.0

    # Creating the matrices that will contains the Rotation matrices components
    Rz = np.zeros(shape = (3,3), dtype=float)
    Ry = np.zeros(shape = (3,3), dtype=float)
    Rx = np.zeros(shape = (3,3), dtype=float)

    # Calculating the raotion matrix parts
    Rz[0][0] = np.cos(alpha)
    Rz[0][1] = -np.sin(alpha)
    Rz[1][0] = np.sin(alpha)
    Rz[1][1] = np.cos(alpha)
    Rz[2][2] = 1

    Ry[0][0] = np.cos(beta)
    Ry[0][2] = np.sin(beta)
    Ry[1][1] = 1
    Ry[2][0] = -np.sin(beta)
    Ry[2][2] = np.cos(beta)

    Rx[0][0] = 1
    Rx[1][1] = np.cos(gamma)
    Rx[1][2] = -np.sin(gamma)
    Rx[2][1] = np.sin(gamma)
    Rx[2][2] = np.cos(gamma)

    return np.matmul(np.matmul(Rz, Ry), Rx)    

###******************************************************************************************************
#FIN
###******************************************************************************************************
    
