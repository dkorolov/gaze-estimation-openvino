import sys
import logging as log
from openvino.inference_engine import IECore

class ModelBase:
    """
    Abstract class for  inference on network model
    """
    IE = None
    net = None
    exec_net = None
    device = None


    def __init__(self, model_xml):
        self.model_xml = model_xml
        self.device = None
        self.IE = IECore()
        self.net = self.IE.read_network(model = model_xml, weights = model_xml.replace('xml', 'bin'))

    def load_model(self, device_name = 'CPU'):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.device = device_name
        if(self.check_model()):
            self.infer_network = self.IE.load_network(network = self.net, device_name = device_name, num_requests = 1)
            self.input_blob = next(iter(self.net.inputs))
            self.output_blob = next(iter(self.net.outputs))
        else:
            log.error("Model have layers unsupported by {} , stop".format(device_name))
            sys.exit(1)

 #   def predict(self, image, req_id):
 #       '''
 #       This method is meant for running predictions on the input image.
 #       '''
 #       input_name = next(iter(self.net.inputs))
 #       input_dict = {input_name:image}
 #       request_handle = self.exec_net.start_async(request_id = req_id, inputs = input_dict)
 #       return request_handle

        
    def check_model(self):
        '''
        Check model for supportes layers.
        '''
        layers_map = self.IE.query_network(network = self.net, device_name = self.device)
        for layer in self.net.layers.keys():
            if layers_map.get(layer, "none") == "none":
                # unsupported layers
                return False
        return True


    def get_input_shape(self):
        # Return the shape of the input layer
        return self.net.inputs[self.input_blob].shape
    

    def exec_net(self, input1, input2=None, input3=None):
        # Start an asynchronous request
        inputiter = iter(self.net.inputs)

        if(input2 is None and input3 is None):
            self.infer_network.start_async(request_id=0, 
                inputs={self.input_blob: input1})
        elif(input3 is None):    
            next(inputiter)#ignore first input
            self.infer_network.start_async(request_id=0, 
                inputs={self.input_blob: input1, next(inputiter):input2})
        else:
            next(inputiter)#ignore first input
            self.infer_network.start_async(request_id=0, 
                inputs={self.input_blob: input1, next(inputiter):input2, next(inputiter):input3})

        return

    def wait(self):
        '''
        Waiting status
        '''
        status = self.infer_network.requests[0].wait(-1)
        return status

    def get_output(self):
        '''
        Get network output
        '''
        return self.infer_network.requests[0].outputs

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        pass

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        pass
