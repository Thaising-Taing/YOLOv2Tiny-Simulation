import os
import torch

from copy import deepcopy as dc

from Weight_Update_Algorithm.Test_with_train import *
from Pre_Processing_Scratch.Pre_Processing import *

from src.RFFP_CUDA_LN import *

class RFFP_CUDA(object):

    def __init__(self, parent):
        self.self           = parent
        self.model          = None
        self.loss           = None
        self.optimizer      = None
        self.scheduler      = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                                        dtype=torch.float32, device='cuda')


    def get_grads(self):
        self.gWeight, self.gBias, self.gGamma, self.gBeta, self.gRunning_Mean_Dec, self.gRunning_Var_Dec = \
            dc(self.Weight), dc(self.Bias), dc(self.Gamma), dc(self.Beta), dc(self.Running_Mean_Dec), dc(self.Running_Var_Dec)
            
        self.gWeight[0]  = self.grads['W0']              
        self.gWeight[1]  = self.grads['W1']              
        self.gWeight[2]  = self.grads['W2']              
        self.gWeight[3]  = self.grads['W3']              
        self.gWeight[4]  = self.grads['W4']              
        self.gWeight[5]  = self.grads['W5']              
        self.gWeight[6]  = self.grads['W6']              
        self.gWeight[7]  = self.grads['W7']              
        self.gWeight[8]  = self.grads['W8']              
        self.gBias       = self.grads['b8']         
        self.gGamma[0]   = self.grads['gamma0']                      
        self.gGamma[1]   = self.grads['gamma1']                      
        self.gGamma[2]   = self.grads['gamma2']                      
        self.gGamma[3]   = self.grads['gamma3']                      
        self.gGamma[4]   = self.grads['gamma4']                      
        self.gGamma[5]   = self.grads['gamma5']                      
        self.gGamma[6]   = self.grads['gamma6']                      
        self.gGamma[7]   = self.grads['gamma7']                      
        self.gBeta[0]    = self.grads['beta0']              
        self.gBeta[1]    = self.grads['beta1']              
        self.gBeta[2]    = self.grads['beta2']              
        self.gBeta[3]    = self.grads['beta3']              
        self.gBeta[4]    = self.grads['beta4']              
        self.gBeta[5]    = self.grads['beta5']              
        self.gBeta[6]    = self.grads['beta6']              
        self.gBeta[7]    = self.grads['beta7']     
        
    def get_weights(self):
        self.Weight[0]              = self.python_model.params['W0']              
        self.Weight[1]              = self.python_model.params['W1']              
        self.Weight[2]              = self.python_model.params['W2']              
        self.Weight[3]              = self.python_model.params['W3']              
        self.Weight[4]              = self.python_model.params['W4']              
        self.Weight[5]              = self.python_model.params['W5']              
        self.Weight[6]              = self.python_model.params['W6']              
        self.Weight[7]              = self.python_model.params['W7']              
        self.Weight[8]              = self.python_model.params['W8']              
        self.Bias                   = self.python_model.params['b8']         
        self.Gamma[0]               = self.python_model.params['gamma0']                      
        self.Gamma[1]               = self.python_model.params['gamma1']                      
        self.Gamma[2]               = self.python_model.params['gamma2']                      
        self.Gamma[3]               = self.python_model.params['gamma3']                      
        self.Gamma[4]               = self.python_model.params['gamma4']                      
        self.Gamma[5]               = self.python_model.params['gamma5']                      
        self.Gamma[6]               = self.python_model.params['gamma6']                      
        self.Gamma[7]               = self.python_model.params['gamma7']                      
        self.Beta[0]                = self.python_model.params['beta0']              
        self.Beta[1]                = self.python_model.params['beta1']              
        self.Beta[2]                = self.python_model.params['beta2']              
        self.Beta[3]                = self.python_model.params['beta3']              
        self.Beta[4]                = self.python_model.params['beta4']              
        self.Beta[5]                = self.python_model.params['beta5']              
        self.Beta[6]                = self.python_model.params['beta6']              
        self.Beta[7]                = self.python_model.params['beta7']              
        self.Running_Mean_Dec[0]    = self.python_model.params['running_mean0']                        
        self.Running_Mean_Dec[1]    = self.python_model.params['running_mean1']                        
        self.Running_Mean_Dec[2]    = self.python_model.params['running_mean2']                        
        self.Running_Mean_Dec[3]    = self.python_model.params['running_mean3']                        
        self.Running_Mean_Dec[4]    = self.python_model.params['running_mean4']                        
        self.Running_Mean_Dec[5]    = self.python_model.params['running_mean5']                        
        self.Running_Mean_Dec[6]    = self.python_model.params['running_mean6']                        
        self.Running_Mean_Dec[7]    = self.python_model.params['running_mean7']                        
        self.Running_Var_Dec[0]     = self.python_model.params['running_var0']                       
        self.Running_Var_Dec[1]     = self.python_model.params['running_var1']                       
        self.Running_Var_Dec[2]     = self.python_model.params['running_var2']                       
        self.Running_Var_Dec[3]     = self.python_model.params['running_var3']                       
        self.Running_Var_Dec[4]     = self.python_model.params['running_var4']                       
        self.Running_Var_Dec[5]     = self.python_model.params['running_var5']                       
        self.Running_Var_Dec[6]     = self.python_model.params['running_var6']                       
        self.Running_Var_Dec[7]     = self.python_model.params['running_var7']                       
        
        
    def load_weights(self, data):
        try: self.Weight, self.Bias, self.Gamma, self.Beta, self.Running_Mean_Dec, self.Running_Var_Dec = data
        except: self.Weight, self.Bias, self.Gamma, self.Beta = data
        self.python_model.params['W0']            = self.Weight[0]
        self.python_model.params['W1']            = self.Weight[1]
        self.python_model.params['W2']            = self.Weight[2]
        self.python_model.params['W3']            = self.Weight[3]
        self.python_model.params['W4']            = self.Weight[4]
        self.python_model.params['W5']            = self.Weight[5]
        self.python_model.params['W6']            = self.Weight[6]
        self.python_model.params['W7']            = self.Weight[7]
        self.python_model.params['W8']            = self.Weight[8]
        self.python_model.params['b8']            = self.Bias
        self.python_model.params['gamma0']        = self.Gamma[0]
        self.python_model.params['gamma1']        = self.Gamma[1]
        self.python_model.params['gamma2']        = self.Gamma[2]
        self.python_model.params['gamma3']        = self.Gamma[3]
        self.python_model.params['gamma4']        = self.Gamma[4]
        self.python_model.params['gamma5']        = self.Gamma[5]
        self.python_model.params['gamma6']        = self.Gamma[6]
        self.python_model.params['gamma7']        = self.Gamma[7]
        self.python_model.params['beta0']         = self.Beta[0]
        self.python_model.params['beta1']         = self.Beta[1]
        self.python_model.params['beta2']         = self.Beta[2]
        self.python_model.params['beta3']         = self.Beta[3]
        self.python_model.params['beta4']         = self.Beta[4]
        self.python_model.params['beta5']         = self.Beta[5]
        self.python_model.params['beta6']         = self.Beta[6]
        self.python_model.params['beta7']         = self.Beta[7]
        self.python_model.params['running_mean0'] = self.Running_Mean_Dec[0]
        self.python_model.params['running_mean1'] = self.Running_Mean_Dec[1]
        self.python_model.params['running_mean2'] = self.Running_Mean_Dec[2]
        self.python_model.params['running_mean3'] = self.Running_Mean_Dec[3]
        self.python_model.params['running_mean4'] = self.Running_Mean_Dec[4]
        self.python_model.params['running_mean5'] = self.Running_Mean_Dec[5]
        self.python_model.params['running_mean6'] = self.Running_Mean_Dec[6]
        self.python_model.params['running_mean7'] = self.Running_Mean_Dec[7]
        self.python_model.params['running_var0']  = self.Running_Var_Dec[0]
        self.python_model.params['running_var1']  = self.Running_Var_Dec[1]
        self.python_model.params['running_var2']  = self.Running_Var_Dec[2]
        self.python_model.params['running_var3']  = self.Running_Var_Dec[3]
        self.python_model.params['running_var4']  = self.Running_Var_Dec[4]
        self.python_model.params['running_var5']  = self.Running_Var_Dec[5]
        self.python_model.params['running_var6']  = self.Running_Var_Dec[6]
        self.python_model.params['running_var7']  = self.Running_Var_Dec[7]

    def Before_Forward(self,Input):
        pass

    def Forward(self, data):
        self.gt_boxes       = data.gt_boxes  
        self.gt_classes     = data.gt_classes
        self.num_boxes      = data.num_obj 
        self.num_obj        = data.num_obj 
        self.image          = data.im_data.cuda()
        X = data.im_data
        self.out, self.cache, self.Out_all_layers = self.python_model.forward(X)
        
            
    def forward_pred(self, out):
        """
        Evaluate loss and gradient for the deep convolutional network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        # print('Calculating the loss and its gradients for pytorch model.')

        scores = out
        bsize, _, h, w = out.shape
        out = out.permute(0, 2, 3, 1).contiguous().view(bsize, 13 * 13 * 5, 5 + 20)
        # Calculate losses based on loss functions(box loss, Intersection over Union(IoU) loss, class loss)
        xy_pred = torch.sigmoid(out[:, :, 0:2]) #
        conf_pred = torch.sigmoid(out[:, :, 4:5]) # 
        hw_pred = torch.exp(out[:, :, 2:4])
        class_score = out[:, :, 5:]
        class_pred = F.softmax(class_score, dim=-1)
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

        # dout = open("./Pytorch_Backward_loss_gradients.pickle", "rb")
        # dout = pickle.load(dout)
        # print('\n\n',dout.dtype, dout[dout!=0])
        return delta_pred, conf_pred, class_pred
        
        
        
    def Calculate_Loss(self,data):
        out = self.out
        self.loss, self.dout = self.python_model.loss(out, self.gt_boxes, self.gt_classes, self.num_boxes)
        
    def Backward(self,data):
        self.dout, self.grads = self.python_model.backward(self.dout, self.cache)
        self.get_weights()
        self.get_grads()