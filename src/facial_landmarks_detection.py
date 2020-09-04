import numpy as np
import cv2
import time
from model_base import ModelBase

class FacialLandmarksDetection(ModelBase):
    '''
    Class for the Face Landmarks Detection Model
    OpenVino pretrained model: landmarks-regression-retail-0009
    '''

    def predict(self, image):
        '''
        Method for running Facial Landmarks predictions on the input image.
        '''
        self.exec_net(image)
        # Wait for the result
        if self.wait() == 0:
            result = (self.get_output())[self.output_blob]
            return result


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        # get input shape - [1x3x48x48]
        net_input_shape = self.get_input_shape()

        p_frame = np.copy(image)
        p_frame = cv2.resize(p_frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        return p_frame
        #raise NotImplementedError

    def preprocess_output(self, outputs, facebox, image, overlay_flag=True, threshold = 0.5):
        '''
        Preprocess data for model inference
        '''

        normed_landmarks = outputs.reshape(1, 10)[0]

        # facebox = (xmin,ymin,xmax,ymax)
        height = facebox[3]-facebox[1] #ymax-ymin
        width = facebox[2]-facebox[0]
        
        # Drawing the eyes circles
        if(overlay_flag):
            for i in range(2):
                x = int(normed_landmarks[i*2] * width)
                y = int(normed_landmarks[i*2+1] * height)
                cv2.circle(image, (facebox[0]+x, facebox[1]+y), 30, (0,255,0), 2)
        
        left_eye_point =[normed_landmarks[0] * width,normed_landmarks[1] * height]
        right_eye_point = [normed_landmarks[2] * width,normed_landmarks[3] * height]
        return image, left_eye_point, right_eye_point

