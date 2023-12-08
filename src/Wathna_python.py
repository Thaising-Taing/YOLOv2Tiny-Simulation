import os
import torch

from Weight_Update_Algorithm.Test_with_train import *
from Pre_Processing_Scratch.Pre_Processing import *

from src.Wathna.python_original import *

class Python(object):

    def __init__(self, parent):
        self.self           = parent
        self.model          = None
        self.loss           = None
        self.optimizer      = None
        self.scheduler      = None
        self.device         = None
        self.train_loader   = None
        
        self.Mode                 = self.self.Mode     
        self.Brain_Floating_Point = self.self.Brain_Floating_Point                     
        self.Exponent_Bits        = self.self.Exponent_Bits             
        self.Mantissa_Bits        = self.self.Mantissa_Bits   


        self.PreProcessing = Pre_Processing(Mode =   self.self.Mode,
                                Brain_Floating_Point =   self.self.Brain_Floating_Point,
                                Exponent_Bits        =   self.self.Exponent_Bits,
                                Mantissa_Bits        =   self.self.Mantissa_Bits)


        self.python_model = DeepConvNet(input_dims=(3, 416, 416),
                                        num_filters=[16, 32, 64, 128, 256, 512, 1024, 1024],
                                        max_pools=[0, 1, 2, 3, 4],
                                        weight_scale='kaiming',
                                        batchnorm=True,
                                        dtype=torch.float32, device='cpu')

    def Before_Forward(self,Input):
        pass

    def Forward(self, data):
        self.gt_boxes       = data.gt_boxes  
        self.gt_classes     = data.gt_classes
        self.num_boxes      = data.num_obj 
        self.num_obj        = data.num_obj 
        self.image          = data.im_data
        
        X = data.im_data
        self.out, self.cache, self.Out_all_layers = self.python_model.forward(X)
        
        
    def Calculate_Loss(self,data):
        out = self.out
        self.loss, self.dout = self.python_model.loss(out, self.gt_boxes, self.gt_classes, self.num_boxes)
        
    def Backward(self,data):
        self.dout, self.grads = self.python_model.backward(self.dout, self.cache)