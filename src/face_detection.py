import time
import numpy as np
import cv2
from model_base import ModelBase

class FaceDetection(ModelBase):
    '''
    Class for the Face Detection Model
    OpenVino pretrained model: face-detection-adas-binary-0001
    '''

    def predict(self, image):
        '''
        Method for running Face Detection predictions on the input image.
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
        # get input shape - [1x3x384x672]
        net_input_shape = self.get_input_shape()

        p_frame = np.copy(image)
        p_frame = cv2.resize(p_frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        return p_frame


    def preprocess_output(self, outputs, image, overlay_flag=True, threshold = 0.5):
        '''
        Preprocess data for model inference
        '''
        height = image.shape[0]
        width = image.shape[1]
        faceboxes = []
        # Drawing the box or boxes(if multiple faces)
        for i in range(len(outputs[0][0])):
            box = outputs[0][0][i]
            confidence = box[2]
            if confidence>threshold:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                if(overlay_flag):
                    # Drawing the box on the image
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                faceboxes.append([xmin, ymin,xmax, ymax])
        return image, faceboxes
