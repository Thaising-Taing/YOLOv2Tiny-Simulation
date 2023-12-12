import os
import torch

from Weight_Update_Algorithm.Test_with_train import *
from Pre_Processing_Scratch.Neural_Network_Operations_LightNorm import *
from Pre_Processing_Scratch.Pre_Processing import *

from GiTae_Functions import *
DEBUG = True

def Save_File(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    
class FPGA(object):
    
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
        
    
        # Code by GiTae 
        self.Weight_Dec, self.Bias_Dec, self.Beta_Dec, self.Gamma_Dec, self.Running_Mean_Dec, self.Running_Var_Dec = self.PreProcessing.WeightLoader()
        self.YOLOv2TinyFPGA = YOLOv2_Tiny_FPGA(\
                        self.Weight_Dec, 
                        self.Bias_Dec, 
                        self.Beta_Dec, 
                        self.Gamma_Dec,
                        self.Running_Mean_Dec, 
                        self.Running_Var_Dec,
                        self) 

    def load_weights(self, values):
        if len(values)>4:
            [self.Weight,     self.Bias,     self.Gamma,     self.Beta,     self.Running_Mean    , self.Running_Var    ] = values
            [self.Weight_Dec, self.Bias_Dec, self.Gamma_Dec, self.Beta_Dec, self.Running_Mean_Dec, self.Running_Var_Dec] = values
        else:
            [self.Weight,     self.Bias,     self.Gamma,     self.Beta     ] = values
            [self.Weight_Dec, self.Bias_Dec, self.Gamma_Dec, self.Beta_Dec ] = values
        
    def Before_Forward(self,data):        
        self.gt_boxes       = data.gt_boxes  
        self.gt_classes     = data.gt_classes
        self.num_boxes      = data.num_obj 
        self.num_obj        = data.num_obj 
        self.image          = data.im_data
        self.im_data        = data.im_data
        
        # self.Weight           = data.Weight_Dec
        # self.Bias             = data.Bias_Dec
        # self.Gamma            = data.Gamma_Dec
        # self.Beta             = data.Beta_Dec
        # self.Running_Mean_Dec = data.Running_Mean_Dec
        # self.Running_Var_Dec  = data.Running_Var_Dec
        
        # data.Weight_Dec       = self.Weight                           
        # data.Bias_Dec         = self.Bias                           
        # data.Gamma_Dec        = self.Gamma                           
        # data.Beta_Dec         = self.Beta                           
        # data.Running_Mean_Dec = self.Running_Mean_Dec                       
        # data.Running_Var_Dec  = self.Running_Var_Dec                       
        
        # Code by GiTae
        self.Image_1_start = time.time() 
    
        s = time.time()
        self.YOLOv2TinyFPGA.Write_Weight(self)       
        e = time.time()
        if DEBUG: print("Write Weight Time : ",e-s)
        
        s = time.time()
        self.YOLOv2TinyFPGA.Write_Image(self)
        e = time.time()
        if DEBUG: print("Write Image Time : ",e-s)   
        
        self.d = Device("0000:08:00.0")
        self.bar = self.d.bar[0]
        self.bar.write(0x0, 0x00000011) # yolo start
        self.bar.write(0x0, 0x00000010) # yolo start low

        self.bar.write(0x8, 0x00000011) # rd addr
        self.bar.write(0x0, 0x00000014) # rd en
        self.bar.write(0x0, 0x00000010) # rd en low

        self.bar.write(0x18, 0x00008001) # axi addr
        self.bar.write(0x14, 0x00000001) # axi rd en
        self.bar.write(0x14, 0x00000000) # axi rd en low
        
    
    def Forward(self, data):
        self.gt_boxes       = data.gt_boxes  
        self.gt_classes     = data.gt_classes
        self.num_boxes      = data.num_obj 
        self.num_obj        = data.num_obj 
        self.image          = data.im_data
        self.im_data        = data.im_data    
              
        if DEBUG: print("Start NPU")
        s = time.time()
        self.Output_Layer8 = self.YOLOv2TinyFPGA.Forward(self)
        # Save_File(self.Output_Layer8, "result/output_of_forward_FPGA")
        e = time.time()
        if DEBUG: print("Forward Process Time : ",e-s)
        # self.change_color_red()
        # return Bias_Grad
        self.out = self.Output_Layer8
        
    def Forward_Inference(self, data):
        self.gt_boxes       = data.gt_boxes  
        self.gt_classes     = data.gt_classes
        self.num_boxes      = data.num_obj 
        self.num_obj        = data.num_obj 
        self.image          = data.im_data
        self.im_data        = data.im_data    
              
        if DEBUG: print("Start NPU")
        s = time.time()
        self.Output_Layer8 = self.YOLOv2TinyFPGA.Forward_Inference(data)
        # Save_File(self.Output_Layer8, "result/output_of_forward_FPGA")
        e = time.time()
        if DEBUG: print("Forward Process Time : ",e-s)
        # self.change_color_red()
        # return Bias_Grad
        self.out = self.Output_Layer8
        
        
    def Calculate_Loss(self,data):
                           
        self.Loss, self.Loss_Gradient = self.YOLOv2TinyFPGA.Post_Processing(data, gt_boxes=self.gt_boxes, gt_classes=self.gt_classes, num_boxes=self.num_obj)
    
    def Before_Backward(self,data):
        pass
        self.YOLOv2TinyFPGA.Pre_Processing_Backward(self, self.Loss_Gradient)

    def Backward(self,data):
        s = time.time()
        if DEBUG: print("Backward Start")
        self.gWeight, self.gBias, self.gBeta, self.gGamma = self.YOLOv2TinyFPGA.Backward(self,self.Loss_Gradient)
        e = time.time()
        if DEBUG: print("Backward Process Time : ",e-s)
        # self.change_color_red()
        

    