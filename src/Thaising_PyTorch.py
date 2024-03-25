import os
import torch

from Weight_Update_Algorithm.Test_with_train import *
from Pre_Processing_Scratch.Neural_Network_Operations_LightNorm import *
from Pre_Processing_Scratch.Pre_Processing import *
    
from Post_Processing_Scratch.Calculate_Loss_2Iterations import *
from Thaising_Python import save_file

def Save_File(_path, data):
    _dir = _path.split('/')[1:-1]
    if len(_dir)>1: _dir = os.path.join(_dir)
    else: _dir = _dir[0]
    if not os.path.isdir(_dir): os.mkdir(_dir)
    
    with open(_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def Create_Directory(directory_path):  
    os.makedirs(directory_path, exist_ok=True)

class TorchSimulation(object):
    
    def __init__(self, parent):
        self.self           = parent
        self.model          = None
        self.loss           = None
        self.optimizer      = None
        self.scheduler      = None
        self.device         = None
        self.train_loader   = None
        
        self.save_debug_data= False
        self.save_txt = False
        
        self.save_debug_data1= False
        self.save_bfloat16 = False

        self.Weight = [[] for _ in range(9)]
        self.Bias = None
        self.Gamma = [[] for _ in range(8)]
        self.Beta = [[] for _ in range(8)]
        self.Running_Mean_Dec = [[] for _ in range(8)]
        self.Running_Var_Dec = [[] for _ in range(8)]
        self.gWeight = [[] for _ in range(9)]
        self.gBias = None
        self.gGamma = [[] for _ in range(8)]
        self.gBeta = [[] for _ in range(8)]
        
        self.params = {}
        
        for i in range(8):
            self.params['W{i}'] = self.Weight[i]
            self.params['running_mean{i}'] = self.Running_Mean_Dec[i]
            self.params['running_var{i}'] = self.Running_Var_Dec[i]
            self.params['gamma{i}'] = self.Gamma[i]
            self.params['beta{i}'] = self.Beta[i]
        self.params['W8'] = self.Weight[8]
        self.params['bias'] = self.Bias
        
        self.Mode                 = self.self.Mode     
        self.Brain_Floating_Point = self.self.Brain_Floating_Point                     
        self.Exponent_Bits        = self.self.Exponent_Bits             
        self.Mantissa_Bits        = self.self.Mantissa_Bits  

        if self.save_bfloat16: 
            self.directory_path_bfloat = f"./Output_Sim_PyTorch_LightNorm_Bfloat16"
        if self.save_debug_data or self.save_debug_data1: 
            self.directory_path = f"./Output_Sim_PyTorch_LightNorm"           

        self.PreProcessing = Pre_Processing(Mode =   self.Mode,
                            Brain_Floating_Point =   self.Brain_Floating_Point,
                            Exponent_Bits        =   self.Exponent_Bits,
                            Mantissa_Bits        =   self.Mantissa_Bits)
        
    def load_weights(self, values):
        if len(values)>4:
            [self.Weight,     self.Bias,     self.Gamma,     self.Beta,     self.Running_Mean    , self.Running_Var    ] = values
            [self.Weight_Dec, self.Bias_Dec, self.Gamma_Dec, self.Beta_Dec, self.Running_Mean_Dec, self.Running_Var_Dec] = values
        else:
            [self.Weight,     self.Bias,     self.Gamma,     self.Beta     ] = values
            [self.Weight_Dec, self.Bias_Dec, self.Gamma_Dec, self.Beta_Dec ] = values
        
        for i in range(8):
            self.params['W{i}'] = self.Weight[i]
            self.params['running_mean{i}'] = self.Running_Mean_Dec[i]
            self.params['running_var{i}'] = self.Running_Var_Dec[i]
            self.params['gamma{i}'] = self.Gamma[i]
            self.params['beta{i}'] = self.Beta[i]
        self.params['W8'] = self.Weight[8]
        self.params['bias'] = self.Bias
        
    def Forward(self, data):
        
        self.gt_boxes       = data.gt_boxes  
        self.gt_classes     = data.gt_classes
        self.num_boxes      = data.num_obj 
        self.num_obj        = data.num_obj 
        self.image          = data.im_data
        self.im_data        = data.im_data
        
        self.gt_boxes = self.gt_boxes.cuda()
        self.gt_classes = self.gt_classes.cuda()
        self.num_boxes = self.num_boxes.cuda()
        self.num_obj = self.num_obj.cuda()
        
        im_data             = self.im_data.cuda()
        Weight_Tensor       = self.Weight_Dec
        Gamma_Tensor        = self.Gamma_Dec
        Beta_Tensor         = self.Beta_Dec
        bias                = self.Bias_Dec
        running_mean        = self.Running_Mean_Dec
        running_var         = self.Running_Var_Dec
        filter_size     = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param_stride2 = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        cache = {}
        temp_Out = {}
        temp_cache = {}
        
        for i in range(8):
            Weight_Tensor[i] = Weight_Tensor[i].cuda()
            Gamma_Tensor[i] = Gamma_Tensor[i].cuda()
            Beta_Tensor[i] = Beta_Tensor[i].cuda()
            running_mean[i] = running_mean[i].cuda()
            running_var[i] = running_var[i].cuda()
        Weight_Tensor[8] = Weight_Tensor[8].cuda()
        bias = bias.cuda()

        # Create the Directory
        if self.save_bfloat16:
            Create_Directory(self.directory_path_bfloat)
        if self.save_debug_data or self.save_debug_data1:
            Create_Directory(self.directory_path)
        
        # Layer0: Conv-BN-ReLU-Pool
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Input_Image", im_data)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Weight_Layer7", Weight_Tensor[7])

        if self.save_txt: save_file("Layer0_Input_Image", im_data, module="Conv", layer_no=0, save_txt=True, phase="Forward")

        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Forward_Input_Image", im_data.to(torch.bfloat16))

        if self.save_debug_data1: Save_File(f"{self.directory_path}/Input_Image", im_data)

        temp_Out[0], temp_cache['0'] = Torch_Conv_Pool.forward(im_data, Weight_Tensor[0], conv_param, pool_param_stride2)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Output_1st_Iter_Layer0", temp_Out[0])
        mean, var = Cal_mean_var.forward(temp_Out[0])
        
        Out0, cache['0'] = Torch_Conv_BatchNorm_ReLU_Pool.forward(im_data, Weight_Tensor[0], Gamma_Tensor[0],
                                                                Beta_Tensor[0], conv_param, running_mean[0], 
                                                                running_var[0], mean, var, self.Mode, pool_param_stride2)
        
        if self.save_txt: save_file("Layer0_Output_Maxpooling", im_data, module="Conv", layer_no=0, save_txt=True, phase="Forward")
        
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Output_2nd_Iter_Layer0", Out0)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Weight_Layer0_Before", Weight_Tensor[0])
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Beta_Layer0_Before", Beta_Tensor[0])
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Gamma_Layer0_Before", Gamma_Tensor[0])

        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Forward_Output_2nd_Iter_Layer0", Out0.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Forward_Weight_Layer0_Before", Weight_Tensor[0].to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Forward_Beta_Layer0_Before", Beta_Tensor[0].to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Forward_Gamma_Layer0_Before", Gamma_Tensor[0].to(torch.bfloat16))

        if self.save_debug_data1: Save_File(f"{self.directory_path}/Weight_Conv_0", Weight_Tensor[0])

        # Layer1: Conv-BN-ReLU-Pool
        temp_Out[1], temp_cache['1'] = Torch_FastConv.forward(Out0, Weight_Tensor[1], conv_param)
        
        if self.save_txt: save_file("Layer1_Output_Conv", im_data, module="Conv", layer_no=0, save_txt=True, phase="Forward")
        
        if self.save_debug_data: Save_File(f"{self.directory_path}/Output_1st_Iter_Layer1", temp_Out[1])
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Output_1st_Iter_Layer1", temp_Out[1].to(torch.bfloat16))
        
        mean, var = Cal_mean_var.forward(temp_Out[1])
        
        Out1, cache['1'] = Torch_Conv_BatchNorm_ReLU_Pool.forward(Out0, Weight_Tensor[1], Gamma_Tensor[1], Beta_Tensor[1],
                                                                conv_param, running_mean[1], running_var[1],
                                                                mean, var, self.Mode, pool_param_stride2)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Output_1st_Iter_Layer1", Out1)
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Forward_Output_1st_Iter_Layer1", Out1.to(torch.bfloat16))

        if self.save_debug_data1: Save_File(f"{self.directory_path}/Weight_Conv_1", Weight_Tensor[1])
        
        # Layer2: Conv-BN-ReLU-Pool
        temp_Out[2], temp_cache['2'] = Torch_FastConv.forward(Out1, Weight_Tensor[2], conv_param)
        
        mean, var = Cal_mean_var.forward(temp_Out[2])
        
        Out2, cache['2'] = Torch_Conv_BatchNorm_ReLU_Pool.forward(Out1, Weight_Tensor[2], Gamma_Tensor[2], Beta_Tensor[2],
                                                                conv_param, running_mean[2], running_var[2],
                                                                mean, var, self.Mode, pool_param_stride2)
        
        if self.save_debug_data1: Save_File(f"{self.directory_path}/Weight_Conv_2", Weight_Tensor[2])

        # Layer3: Conv-BN-ReLU-Pool
        temp_Out[3], temp_cache['3'] = Torch_FastConv.forward(Out2, Weight_Tensor[3], conv_param)
        
        mean, var = Cal_mean_var.forward(temp_Out[3])
        
        Out3, cache['3'] = Torch_Conv_BatchNorm_ReLU_Pool.forward(Out2, Weight_Tensor[3], Gamma_Tensor[3], Beta_Tensor[3],
                                                                conv_param, running_mean[3], running_var[3],
                                                                mean, var, self.Mode, pool_param_stride2)
        
        if self.save_debug_data1: Save_File(f"{self.directory_path}/Weight_Conv_3", Weight_Tensor[3])

        # Layer4: Conv-BN-ReLU-Pool
        temp_Out[4], temp_cache['4'] = Torch_FastConv.forward(Out3, Weight_Tensor[4], conv_param)
        
        mean, var = Cal_mean_var.forward(temp_Out[4])
        
        Out4, cache['4'] = Torch_Conv_BatchNorm_ReLU_Pool.forward(Out3, Weight_Tensor[4], Gamma_Tensor[4], Beta_Tensor[4],
                                                                conv_param, running_mean[4], running_var[4],
                                                                mean, var, self.Mode, pool_param_stride2)
        
        if self.save_debug_data1: Save_File(f"{self.directory_path}/Weight_Conv_4", Weight_Tensor[4])

        # Layer5: Conv-BN-ReLU
        temp_Out[5], temp_cache['5'] = Torch_FastConv.forward(Out4, Weight_Tensor[5], conv_param)
        
        mean, var = Cal_mean_var.forward(temp_Out[5])
        
        Out5, cache['5'] = Torch_Conv_BatchNorm_ReLU.forward(Out4, Weight_Tensor[5], Gamma_Tensor[5], Beta_Tensor[5],
                                                            conv_param, running_mean[5], running_var[5],
                                                            mean, var, self.Mode)
        
        if self.save_debug_data1: Save_File(f"{self.directory_path}/Weight_Conv_5", Weight_Tensor[5])

        # Layer6: Conv-BN-ReLU
        temp_Out[6], temp_cache['6'] = Torch_FastConv.forward(Out5, Weight_Tensor[6], conv_param)
        
        mean, var = Cal_mean_var.forward(temp_Out[6])
        
        Out6, cache['6'] = Torch_Conv_BatchNorm_ReLU.forward(Out5, Weight_Tensor[6], Gamma_Tensor[6],
                                                            Beta_Tensor[6], conv_param, running_mean[6], running_var[6],
                                                            mean, var, self.Mode)
        
        if self.save_debug_data1: Save_File(f"{self.directory_path}/Weight_Conv_6", Weight_Tensor[6])

        # Layer7: Conv-BN-ReLU
        temp_Out[7], temp_cache['7'] = Torch_FastConv.forward(Out6, Weight_Tensor[7], conv_param)
        
        mean, var = Cal_mean_var.forward(temp_Out[7])
        
        Out7, cache['7'] = Torch_Conv_BatchNorm_ReLU.forward(Out6, Weight_Tensor[7], Gamma_Tensor[7], Beta_Tensor[7],
                                                            conv_param, running_mean[7], running_var[7],
                                                            mean, var, self.Mode)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Input_Layer7", Out6)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Weight_Layer7", Weight_Tensor[7])
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Gamma_Layer7", Gamma_Tensor[7])
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Beta_Layer7", Beta_Tensor[7])
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_running_mean_Layer7", running_mean[7])
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_running_var_Layer7", running_var[7])
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Output_Layer7", Out7)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Cache_Layer7", cache['7'])

        if self.save_debug_data1: Save_File(f"{self.directory_path}/Weight_Conv_7", Weight_Tensor[7])

        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Forward_Input_Layer7", Out6.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Forward_Weight_Layer7", Weight_Tensor[7].to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Forward_Gamma_Layer7", Gamma_Tensor[7].to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Forward_Beta_Layer7", Beta_Tensor[7].to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Forward_running_mean_Layer7", running_mean[7].to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Forward_running_var_Layer7", running_var[7].to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Forward_Output_Layer7", Out7.to(torch.bfloat16))

        # Layer8: ConvWB
        conv_param['pad'] = 0
        Out8, cache['8'] = Torch_FastConvWB.forward(Out7, Weight_Tensor[8], bias, conv_param)
        if self.save_txt: save_file("Weight_Layer8", Weight_Tensor[8], module="Conv", layer_no=0, save_txt=True, phase="Forward")
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Input_Layer8", Out7)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Weight_Layer8", Weight_Tensor[8])
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Bias", bias)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Output_Layer8", Out8)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Forward_Cache_Layer8", cache['8'])

        if self.save_debug_data1: Save_File(f"{self.directory_path}/Weight_Conv_8", Weight_Tensor[8])

        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Forward_Input_Layer8", Out7.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Forward_Weight_Layer8", Weight_Tensor[8].to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Forward_Bias", bias.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Forward_Output_Layer8", Out8.to(torch.bfloat16))

        if self.save_debug_data1: Save_File(f"{self.directory_path}/Output_Forward", Out8)

        self.Output_Image, self.cache = Out8, cache 
        self.out = Out8
        # return Output_Image, cache
        
    def Calculate_Loss(self,data):
        self.Loss, self.Loss_Gradient = loss(out=self.Output_Image, gt_boxes=self.gt_boxes, gt_classes=self.gt_classes, num_boxes=self.num_boxes)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Loss_Grad", self.Loss_Gradient)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Loss", self.Loss)

        if self.save_debug_data1: Save_File(f"{self.directory_path}/Loss_Gradient", self.Loss_Gradient)

    def Backward(self,data):
        # Add By Thaising
        Loss_Gradient, cache = self.Loss_Gradient, self.cache
        Input_Grad_Layer8, Weight_Gradient_Layer8, Bias_Grad  = Torch_FastConvWB.backward(Loss_Gradient, cache['8'])
        if self.save_txt: save_file("Loss_Gradient", Loss_Gradient, module="Conv", layer_no=0, save_txt=True, phase="Forward")
        if self.save_txt: save_file("Input_Grad_Layer8", Input_Grad_Layer8, module="Conv", layer_no=0, save_txt=True, phase="Forward")
        if self.save_txt: save_file("Weight_Gradient_Layer8", Weight_Gradient_Layer8, module="Conv", layer_no=0, save_txt=True, phase="Forward")
        if self.save_txt: save_file("Bias_Grad", Bias_Grad, module="Conv", layer_no=0, save_txt=True, phase="Forward")
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Loss_Gradient_Layer8", Loss_Gradient)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Input_Gradient_Layer8", Input_Grad_Layer8)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Weight_Gradient_Layer8", Weight_Gradient_Layer8)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Bias_Gradient_Layer8", Bias_Grad)

        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_8_Backward_Input_Gradient", Input_Grad_Layer8)
        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_8_Backward_Weight_Gradient", Weight_Gradient_Layer8)

        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Loss_Gradient_Layer8", Loss_Gradient.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Input_Gradient_Layer8", Input_Grad_Layer8.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Weight_Gradient_Layer8", Weight_Gradient_Layer8.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Bias_Gradient_Layer8", Bias_Grad.to(torch.bfloat16))

        Input_Grad_Layer7, Weight_Gradient_Layer7, Gamma_Gradient_Layer7, Beta_Gradient_Layer7  = Torch_Conv_BatchNorm_ReLU.backward (Input_Grad_Layer8, cache['7'])
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Input_Gradient_Layer7", Input_Grad_Layer7)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Weight_Gradient_Layer7", Weight_Gradient_Layer7)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Gamma_Gradient_Layer7", Gamma_Gradient_Layer7)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Beta_Gradient_Layer7", Beta_Gradient_Layer7)

        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_7_Backward_Input_Gradient", Input_Grad_Layer7)
        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_7_Backward_Weight_Gradient", Weight_Gradient_Layer7)

        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Input_Gradient_Layer7", Input_Grad_Layer7.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Weight_Gradient_Layer7", Weight_Gradient_Layer7.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Gamma_Gradient_Layer7", Gamma_Gradient_Layer7.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Beta_Gradient_Layer7", Beta_Gradient_Layer7.to(torch.bfloat16))

        Input_Grad_Layer6, Weight_Gradient_Layer6, Gamma_Gradient_Layer6, Beta_Gradient_Layer6  = Torch_Conv_BatchNorm_ReLU.backward (Input_Grad_Layer7, cache['6'])
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Input_Gradient_Layer6", Input_Grad_Layer6)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Weight_Gradient_Layer6", Weight_Gradient_Layer6)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Gamma_Gradient_Layer6", Gamma_Gradient_Layer6)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Beta_Gradient_Layer6", Beta_Gradient_Layer6)

        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_6_Backward_Input_Gradient", Input_Grad_Layer6)
        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_6_Backward_Weight_Gradient", Weight_Gradient_Layer6)
        
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Input_Gradient_Layer6", Input_Grad_Layer6.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Weight_Gradient_Layer6", Weight_Gradient_Layer6.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Gamma_Gradient_Layer6", Gamma_Gradient_Layer6.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Beta_Gradient_Layer6", Beta_Gradient_Layer6.to(torch.bfloat16))        
        
        Input_Grad_Layer5, Weight_Gradient_Layer5, Gamma_Gradient_Layer5, Beta_Gradient_Layer5  = Torch_Conv_BatchNorm_ReLU.backward (Input_Grad_Layer6, cache['5'])
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Input_Gradient_Layer5", Input_Grad_Layer5)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Weight_Gradient_Layer5", Weight_Gradient_Layer5)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Gamma_Gradient_Layer5", Gamma_Gradient_Layer5)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Beta_Gradient_Layer5", Beta_Gradient_Layer5)

        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_5_Backward_Input_Gradient", Input_Grad_Layer5)
        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_5_Backward_Weight_Gradient", Weight_Gradient_Layer5)
        
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Input_Gradient_Layer5", Input_Grad_Layer5.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Weight_Gradient_Layer5", Weight_Gradient_Layer5.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Gamma_Gradient_Layer5", Gamma_Gradient_Layer5.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Beta_Gradient_Layer5", Beta_Gradient_Layer5.to(torch.bfloat16))     

        Input_Grad_Layer4, Weight_Gradient_Layer4, Gamma_Gradient_Layer4, Beta_Gradient_Layer4  = Torch_Conv_BatchNorm_ReLU_Pool.backward (Input_Grad_Layer5, cache['4'])
        if self.save_txt: save_file("Weight_Gradient_Layer4", Weight_Gradient_Layer4, module="Conv", layer_no=0, save_txt=True, phase="Forward")
        
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Input_Gradient_Layer4", Input_Grad_Layer4)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Weight_Gradient_Layer4", Weight_Gradient_Layer4)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Gamma_Gradient_Layer4", Gamma_Gradient_Layer4)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Beta_Gradient_Layer4", Beta_Gradient_Layer4)

        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_4_Backward_Input_Gradient", Input_Grad_Layer4)
        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_4_Backward_Weight_Gradient", Weight_Gradient_Layer4)
        
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Input_Gradient_Layer4", Input_Grad_Layer4.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Weight_Gradient_Layer4", Weight_Gradient_Layer4.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Gamma_Gradient_Layer4", Gamma_Gradient_Layer4.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Beta_Gradient_Layer4", Beta_Gradient_Layer4.to(torch.bfloat16)) 
        
        Input_Grad_Layer3, Weight_Gradient_Layer3, Gamma_Gradient_Layer3, Beta_Gradient_Layer3  = Torch_Conv_BatchNorm_ReLU_Pool.backward (Input_Grad_Layer4, cache['3'])
        if self.save_txt: save_file("Weight_Gradient_Layer3", Weight_Gradient_Layer3, module="Conv", layer_no=0, save_txt=True, phase="Forward")
        
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Input_Grad_Layer3", Input_Grad_Layer3)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Weight_Gradient_Layer3", Weight_Gradient_Layer3)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Gamma_Gradient_Layer3", Gamma_Gradient_Layer3)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Beta_Gradient_Layer3", Beta_Gradient_Layer3)   

        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_3_Backward_Input_Gradient", Input_Grad_Layer3)
        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_3_Backward_Weight_Gradient", Weight_Gradient_Layer3)

        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Input_Grad_Layer3", Input_Grad_Layer3.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Weight_Gradient_Layer3", Weight_Gradient_Layer3.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Gamma_Gradient_Layer3", Gamma_Gradient_Layer3.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Beta_Gradient_Layer3", Beta_Gradient_Layer3.to(torch.bfloat16))     

        Input_Grad_Layer2, Weight_Gradient_Layer2, Gamma_Gradient_Layer2, Beta_Gradient_Layer2  = Torch_Conv_BatchNorm_ReLU_Pool.backward (Input_Grad_Layer3, cache['2'])
        if self.save_txt: save_file("Weight_Gradient_Layer2", Weight_Gradient_Layer2, module="Conv", layer_no=0, save_txt=True, phase="Forward")
        
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Input_Grad_Layer2", Input_Grad_Layer2)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Weight_Gradient_Layer2", Weight_Gradient_Layer2)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Gamma_Gradient_Layer2", Gamma_Gradient_Layer2)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Beta_Gradient_Layer2", Beta_Gradient_Layer2)

        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_2_Backward_Input_Gradient", Input_Grad_Layer2)
        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_2_Backward_Weight_Gradient", Weight_Gradient_Layer2)          

        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Input_Grad_Layer2.pickle", Input_Grad_Layer2.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Weight_Gradient_Layer2", Weight_Gradient_Layer2.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Gamma_Gradient_Layer2", Gamma_Gradient_Layer2.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Beta_Gradient_Layer2", Beta_Gradient_Layer2.to(torch.bfloat16))  

        Input_Grad_Layer1, Weight_Gradient_Layer1, Gamma_Gradient_Layer1, Beta_Gradient_Layer1  = Torch_Conv_BatchNorm_ReLU_Pool.backward (Input_Grad_Layer2, cache['1'])
        if self.save_txt: save_file("Weight_Gradient_Layer1", Weight_Gradient_Layer1, module="Conv", layer_no=0, save_txt=True, phase="Forward")
        
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Input_Grad_Layer1", Input_Grad_Layer1)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Weight_Gradient_Layer1", Weight_Gradient_Layer1)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Gamma_Gradient_Layer1", Gamma_Gradient_Layer1)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Beta_Gradient_Layer1", Beta_Gradient_Layer1)

        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_1_Backward_Input_Gradient", Input_Grad_Layer1)
        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_1_Backward_Weight_Gradient", Weight_Gradient_Layer1)

        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Input_Grad_Layer1", Input_Grad_Layer1.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Weight_Gradient_Layer1", Weight_Gradient_Layer1.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Gamma_Gradient_Layer1", Gamma_Gradient_Layer1.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Beta_Gradient_Layer1", Beta_Gradient_Layer1.to(torch.bfloat16))

        Input_Grad_Layer0, Weight_Gradient_Layer0, Gamma_Gradient_Layer0, Beta_Gradient_Layer0  = Torch_Conv_BatchNorm_ReLU_Pool.backward (Input_Grad_Layer1, cache['0'])
        if self.save_txt: save_file("Weight_Gradient_Layer0", Weight_Gradient_Layer0, module="Conv", layer_no=0, save_txt=True, phase="Forward")
        
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Input_Grad_Layer0", Input_Grad_Layer0)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Weight_Gradient_Layer0", Weight_Gradient_Layer0)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Gamma_Gradient_Layer0", Gamma_Gradient_Layer0)
        if self.save_debug_data: Save_File(f"{self.directory_path}/Backward_Beta_Gradient_Layer0", Beta_Gradient_Layer0)

        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_0_Backward_Input_Gradient", Input_Grad_Layer0)
        if self.save_debug_data1: Save_File(f"{self.directory_path}/Layer_0_Backward_Weight_Gradient", Weight_Gradient_Layer0)
        
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Input_Grad_Layer0", Input_Grad_Layer0.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Weight_Gradient_Layer0", Weight_Gradient_Layer0.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Gamma_Gradient_Layer0", Gamma_Gradient_Layer0.to(torch.bfloat16))
        if self.save_bfloat16: Save_File(f"{self.directory_path_bfloat}/Backward_Beta_Gradient_Layer0", Beta_Gradient_Layer0.to(torch.bfloat16))
       
        # Gradient Value for Weight Update
        self.gWeight = [Weight_Gradient_Layer0.cuda(), Weight_Gradient_Layer1.cuda(), Weight_Gradient_Layer2.cuda(), Weight_Gradient_Layer3.cuda(), 
                        Weight_Gradient_Layer4.cuda(), Weight_Gradient_Layer5.cuda(), Weight_Gradient_Layer6.cuda(), Weight_Gradient_Layer7.cuda(), 
                        Weight_Gradient_Layer8.cuda()]
        
        self.gBias  = Bias_Grad.cuda()
        
        self.gGamma = [Gamma_Gradient_Layer0.cuda(), Gamma_Gradient_Layer1.cuda(), Gamma_Gradient_Layer2.cuda(), Gamma_Gradient_Layer3.cuda(), 
                        Gamma_Gradient_Layer4.cuda(), Gamma_Gradient_Layer5.cuda(), Gamma_Gradient_Layer6.cuda(), Gamma_Gradient_Layer7.cuda()]
        
        self.gBeta  = [Beta_Gradient_Layer0.cuda(), Beta_Gradient_Layer1.cuda(), Beta_Gradient_Layer2.cuda(), Beta_Gradient_Layer3.cuda(), 
                        Beta_Gradient_Layer4.cuda(), Beta_Gradient_Layer5.cuda(),Beta_Gradient_Layer6.cuda(), Beta_Gradient_Layer7.cuda()]