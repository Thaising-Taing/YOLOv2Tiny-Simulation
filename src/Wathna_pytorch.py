import os
import torch

from Weight_Update_Algorithm.Test_with_train import *
from Pre_Processing_Scratch.Pre_Processing import *

from src.Wathna.torch_2iteration import *
    
class Pytorch(object):
    
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
        

        self.modtorch_model = DeepConvNetTorch(input_dims=(3, 416, 416),
                                        num_filters=[16, 32, 64, 128, 256, 512, 1024, 1024],
                                        max_pools=[0, 1, 2, 3, 4],
                                        weight_scale='kaiming',
                                        batchnorm=True,
                                        dtype=torch.float32, device='cpu')

    def load_weights(self, data):
        Weight, Bias, Gamma_WeightBN, BetaBN, Running_Mean_Dec, Running_Var_Dec = data
        self.modtorch_model.params['W0']            = Weight[0]
        self.modtorch_model.params['W1']            = Weight[1]
        self.modtorch_model.params['W2']            = Weight[2]
        self.modtorch_model.params['W3']            = Weight[3]
        self.modtorch_model.params['W4']            = Weight[4]
        self.modtorch_model.params['W5']            = Weight[5]
        self.modtorch_model.params['W6']            = Weight[6]
        self.modtorch_model.params['W7']            = Weight[7]
        self.modtorch_model.params['W8']            = Weight[8]
        self.modtorch_model.params['b8']            = Bias
        self.modtorch_model.params['gamma0']        = Gamma_WeightBN[0]
        self.modtorch_model.params['gamma1']        = Gamma_WeightBN[1]
        self.modtorch_model.params['gamma2']        = Gamma_WeightBN[2]
        self.modtorch_model.params['gamma3']        = Gamma_WeightBN[3]
        self.modtorch_model.params['gamma4']        = Gamma_WeightBN[4]
        self.modtorch_model.params['gamma5']        = Gamma_WeightBN[5]
        self.modtorch_model.params['gamma6']        = Gamma_WeightBN[6]
        self.modtorch_model.params['gamma7']        = Gamma_WeightBN[7]
        self.modtorch_model.params['beta0']         = BetaBN[0]
        self.modtorch_model.params['beta1']         = BetaBN[1]
        self.modtorch_model.params['beta2']         = BetaBN[2]
        self.modtorch_model.params['beta3']         = BetaBN[3]
        self.modtorch_model.params['beta4']         = BetaBN[4]
        self.modtorch_model.params['beta5']         = BetaBN[5]
        self.modtorch_model.params['beta6']         = BetaBN[6]
        self.modtorch_model.params['beta7']         = BetaBN[7]
        self.modtorch_model.params['running_mean0'] = Running_Mean_Dec[0]
        self.modtorch_model.params['running_mean1'] = Running_Mean_Dec[1]
        self.modtorch_model.params['running_mean2'] = Running_Mean_Dec[2]
        self.modtorch_model.params['running_mean3'] = Running_Mean_Dec[3]
        self.modtorch_model.params['running_mean4'] = Running_Mean_Dec[4]
        self.modtorch_model.params['running_mean5'] = Running_Mean_Dec[5]
        self.modtorch_model.params['running_mean6'] = Running_Mean_Dec[6]
        self.modtorch_model.params['running_mean7'] = Running_Mean_Dec[7]
        self.modtorch_model.params['running_var0']  = Running_Var_Dec[0]
        self.modtorch_model.params['running_var1']  = Running_Var_Dec[1]
        self.modtorch_model.params['running_var2']  = Running_Var_Dec[2]
        self.modtorch_model.params['running_var3']  = Running_Var_Dec[3]
        self.modtorch_model.params['running_var4']  = Running_Var_Dec[4]
        self.modtorch_model.params['running_var5']  = Running_Var_Dec[5]
        self.modtorch_model.params['running_var6']  = Running_Var_Dec[6]
        self.modtorch_model.params['running_var7']  = Running_Var_Dec[7]

    def Before_Forward(self,Input):
            pass
        
        
         
    def Forward(self, data):
        self.gt_boxes       = data.gt_boxes  
        self.gt_classes     = data.gt_classes
        self.num_boxes      = data.num_obj 
        self.num_obj        = data.num_obj 
        self.image          = data.im_data
        
        X = data.im_data
        self.out, self.cache, self.Out_all_layers = self.modtorch_model.forward(X)
        
    def Calculate_Loss(self,data):
        out = self.out
        self.loss, self.dout = self.modtorch_model.loss(out, self.gt_boxes, self.gt_classes, self.num_boxes)
        
    def Backward(self,data):
        self.dout, self.grads = self.modtorch_model.backward(self.dout, self.cache)