import cv2
import numpy as np
import time
from model_base import ModelBase


class GazeEstimation(ModelBase):
    '''
    Class for the Gaze Estimation Model
    OpenVino pretrained model: gaze-estimation-adas-0002
    '''

    def predict(self, left_eye_image, right_eye_image, headpose_angles):
        '''
        Method for running gaze predictions on the input image.
        '''
        self.exec_net(headpose_angles, left_eye_image,right_eye_image)

        # Wait for the result
        if self.wait() == 0:
            # end time of inference
            end_time = time.time()
            result = (self.get_output())[self.output_blob]
            return result


    def preprocess_input(self, frame, face, left_eye_point, right_eye_point, overlay_flag=True):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.

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

        '''
        
        lefteye_input_shape =  [1,3,60,60] #self.get_input_shape()
        righteye_input_shape = [1,3,60,60] #self.get_next_input_shape(2)

        # crop left eye
        x_center = left_eye_point[0]
        y_center = left_eye_point[1]
        width = lefteye_input_shape[3]
        height = lefteye_input_shape[2]
        # ymin:ymax, xmin:xmax 
        facewidthedge = face.shape[1]
        faceheightedge = face.shape[0]
        
        # check for edges to not crop
        ymin = int(y_center - height//2) if  int(y_center - height//2) >=0 else 0 
        ymax = int(y_center + height//2) if  int(y_center + height//2) <=faceheightedge else faceheightedge

        xmin = int(x_center - width//2) if  int(x_center - width//2) >=0 else 0 
        xmax = int(x_center + width//2) if  int(x_center + width//2) <=facewidthedge else facewidthedge

        # left eye [1x3x60x60]
        left_eye_image = face[ymin: ymax, xmin:xmax]
        p_frame_left = cv2.resize(left_eye_image, (lefteye_input_shape[3], lefteye_input_shape[2]))
        p_frame_left = p_frame_left.transpose((2,0,1))
        p_frame_left = p_frame_left.reshape(1, *p_frame_left.shape)

        # crop right eye
        x_center = right_eye_point[0]
        y_center = right_eye_point[1]
        width = righteye_input_shape[3]
        height = righteye_input_shape[2]
        # ymin:ymax, xmin:xmax 
        # check for edges to not crop
        ymin = int(y_center - height//2) if  int(y_center - height//2) >=0 else 0 
        ymax = int(y_center + height//2) if  int(y_center + height//2) <=faceheightedge else faceheightedge

        xmin = int(x_center - width//2) if  int(x_center - width//2) >=0 else 0 
        xmax = int(x_center + width//2) if  int(x_center + width//2) <=facewidthedge else facewidthedge

        # right eye [1x3x60x60]
        right_eye_image =  face[ymin: ymax, xmin:xmax]
        p_frame_right = cv2.resize(right_eye_image, (righteye_input_shape[3], righteye_input_shape[2]))
        p_frame_right = p_frame_right.transpose((2,0,1))
        p_frame_right = p_frame_right.reshape(1, *p_frame_right.shape)

        return frame, p_frame_left, p_frame_right


    def preprocess_output(self, outputs, image,facebox, left_eye_point, right_eye_point,overlay_flag=True, threshold = 0.5):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

        The net outputs a blob with the shape: [1, 3], containing Cartesian coordinates of gaze direction vector. Please note that the output vector is not normalizes and has non-unit length.
        Output layer name in Inference Engine format:
        gaze_vector
        '''
        x = outputs[0][0]
        y = outputs[0][1]
        z = outputs[0][2]

        #Draw output
        if(overlay_flag):
            #left eye
            xmin,ymin,_,_ = facebox
            x_center = left_eye_point[0]
            y_center = left_eye_point[1]
            left_eye_center_x = int(xmin + x_center)
            left_eye_center_y = int(ymin + y_center)
            #right eye
            x_center = right_eye_point[0]
            y_center = right_eye_point[1]
            right_eye_center_x = int(xmin + x_center)
            right_eye_center_y = int(ymin + y_center)

            cv2.putText(image,"Left eye position:  X: {:.1f}  Y: {:.1f} ".format(left_eye_center_x, left_eye_center_y) , (20, 70), 0,0.6, (0,220,0), 1)
            cv2.putText(image,"Right eye position:  X: {:.1f}  Y: {:.1f} ".format(right_eye_center_x, right_eye_center_y) , (20, 90), 0,0.6, (0,220,0), 1)
            cv2.putText(image,"Gaze Direction Vector :" , (20, 120), 0,0.6, (0,220,0), 1)
            cv2.putText(image,"Axe X: {:.1f}  Axe Y: {:.1f}  Axe Z: {:.1f}".format(x*100, y*100, z) , (20, 140), 0,0.6, (0,220,0), 1)

            cv2.arrowedLine(image, (left_eye_center_x,left_eye_center_y), (left_eye_center_x + int(x*100),left_eye_center_y + int(-y*100)), (255, 50, 50), 5)
            cv2.arrowedLine(image, (right_eye_center_x,right_eye_center_y), (right_eye_center_x + int(x*100),right_eye_center_y + int(-y*100)), (255,50, 50), 5)

        return image, [x, y, z]
