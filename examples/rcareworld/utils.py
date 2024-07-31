import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

def unity2storm(position=None, orientation=None):
    # transformation from Storm to Unity is : then mirror img about xy axis and then rotate -90 deg about x
    # (vice versa also works somehow) (must mean mirror img and axis rotation are commutative for this case)
    # TODO Pranav : verify this
    unity2storm_rot = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    
    x_neg90_rot_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    x_neg90_rot_quat = Quaternion(axis=[1, 0, 0], angle=-1.5708) #same as above matrix but in quaternion form, convention : wxyz

    if position==None and not orientation==None:
        
        # Orientation input has to be in Euler angles form, output will be quaternion
        
        # Rotations are intrinsic since they are around body axes , so capital letters. 
        # Source : https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html

        # Rotation convention taken from here:
        # https://docs.unity3d.com/ScriptReference/Transform-eulerAngles.html 

        orientation[2] = -1 * orientation[2]
        
        # print(R.from_euler('ZXY', orientation, degrees=True).as_matrix())
        q = Quaternion(matrix=R.from_euler('ZXY', orientation, degrees=True).as_matrix())

        # pyquaternion works xyzw form. So no need for reindexing
        g_q = x_neg90_rot_quat * q * x_neg90_rot_quat.inverse

        return np.ravel([g_q[0], g_q[1], g_q[2], g_q[3]])

    elif not position==None and orientation==None:
    
        # For position vector, the mirror image and rotation can be condensed into one matrix.
        g_pos = np.matmul(unity2storm_rot,position)
        return g_pos

    else:
        print("Both/Neither position & orientation entered, check again!")