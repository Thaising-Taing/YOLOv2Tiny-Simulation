import os
import torch

from Weight_Update_Algorithm.Test_with_train import *
from Pre_Processing_Scratch.Neural_Network_Operations_LightNorm import *
from Pre_Processing_Scratch.Pre_Processing import *

from GiTae_Functions import *
DEBUG = True
DEBUG2 = True

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

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
            self.params['W{}'.format(i)] = self.Weight[i]
            self.params['running_mean{}'.format(i)] = self.Running_Mean_Dec[i]
            self.params['running_var{}'.format(i)] = self.Running_Var_Dec[i]
            self.params['gamma{}'.format(i)] = self.Gamma[i]
            self.params['beta{}'.format(i)] = self.Beta[i]
        self.params['W8'] = self.Weight[8]
        self.params['b8'] = self.Bias

        self.optimizer_config = {}
        optim_config = {'learning_rate': 0.01, 'momentum': 0.9}
        for p, _ in self.params.items():
            d = {k: v for k, v in optim_config.items()}
            self.optimizer_config[p] = d
        
        
        
        self.Mode                 = parent.Mode     
        self.Brain_Floating_Point = parent.Brain_Floating_Point                     
        self.Exponent_Bits        = parent.Exponent_Bits             
        self.Mantissa_Bits        = parent.Mantissa_Bits   
        
        
        self.PreProcessing = Pre_Processing(Mode =   self.self.Mode,
                            Brain_Floating_Point =   self.self.Brain_Floating_Point,
                            Exponent_Bits        =   self.self.Exponent_Bits,
                            Mantissa_Bits        =   self.self.Mantissa_Bits)
        
    
        # Code by GiTae 
        self.Weight_Dec, self.Bias_Dec, self.Beta_Dec, self.Gamma_Dec, self.Running_Mean_Dec, self.Running_Var_Dec = self.PreProcessing.WeightLoader()
        self.Weight_Dec         = [x.cuda() for x in self.Weight_Dec       ]
        self.Bias_Dec           = [x.cuda() for x in self.Bias_Dec         ]
        self.Beta_Dec           = [x.cuda() for x in self.Beta_Dec         ]
        self.Gamma_Dec          = [x.cuda() for x in self.Gamma_Dec        ]
        self.Running_Mean_Dec   = [x.cuda() for x in self.Running_Mean_Dec ]
        self.Running_Var_Dec    = [x.cuda() for x in self.Running_Var_Dec  ]
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

        self.gt_boxes = self.gt_boxes.cuda()
        self.gt_classes = self.gt_classes.cuda()
        self.num_boxes = self.num_boxes.cuda()
        self.num_obj = self.num_obj.cuda()
        self.image = self.image.cuda()
        self.im_data = self.im_data.cuda()

        
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
        
                
        
        # self.d = Device("0000:08:00.0")
        # self.bar = self.d.bar[0]
        # self.bar.write(0x0, 0x00000011) # yolo start
        # self.bar.write(0x0, 0x00000010) # yolo start low

        # self.bar.write(0x8, 0x00000011) # rd addr
        # self.bar.write(0x0, 0x00000014) # rd en
        # self.bar.write(0x0, 0x00000010) # rd en low

        # self.bar.write(0x18, 0x00008001) # axi addr
        # self.bar.write(0x14, 0x00000001) # axi rd en
        # self.bar.write(0x14, 0x00000000) # axi rd en low
        
    
    def Forward(self, data):
        self.parent         = data
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
        self.image = self.image.cuda()
        self.im_data = self.im_data.cuda()
              
        if DEBUG: print("Start NPU")
        
        
        cmd = 'rm -rf src/GiTae/interrupt*txt; touch src/GiTae/interrupt_old.txt; touch src/GiTae/interrupt.txt; python3 src/GiTae/interrupt_layer_0.py '
        os.system(cmd)
        if DEBUG: print(f"Got interrupt")
        
        
        s = time.time()
        self.Output_Layer8 = self.YOLOv2TinyFPGA.Forward(self)
        self.Output_Layer8 = self.Output_Layer8.cuda()
        # Save_File(self.Output_Layer8, "result/output_of_forward_FPGA")
        e = time.time()
        if DEBUG: print("Forward Process Time : ",e-s)
        # self.change_color_red()
        # return Bias_Grad
        self.out = self.Output_Layer8.cuda()
        
    def Forward_Inference(self, data):
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
        self.image = self.image.cuda()
        self.im_data = self.im_data.cuda()
              
        if DEBUG: print("Start NPU")
        
        cmd = 'rm -rf src/GiTae/interrupt*txt; touch src/GiTae/interrupt_old.txt; touch src/GiTae/interrupt.txt; python3 src/GiTae/start_signal.py '
        # cmd = 'python3 src/GiTae/start_signal.py '
        os.system(cmd)
        if DEBUG: print(f"Start signal sent.")
        
        s = time.time()
        # self.Output_Layer8 = self.YOLOv2TinyFPGA.Forward_Inference(data)
        self.Output_Layer8 = self.YOLOv2TinyFPGA.Forward_Infer(self)
        Save_File(self.Output_Layer8, "result/output_of_forward_FPGA")
        e = time.time()
        if DEBUG: print("Forward Process Time : ",e-s)
        # self.change_color_red()
        # return Bias_Grad
        self.out = self.Output_Layer8
        
    def Calculate_Loss(self,data):
        self.gt_boxes    = self.gt_boxes.cuda()
        self.gt_classes  = self.gt_classes.cuda()
        self.num_obj     = self.num_obj.cuda()
        
        self.Loss, self.Loss_Gradient = self.YOLOv2TinyFPGA.Post_Processing(data, gt_boxes=self.gt_boxes, gt_classes=self.gt_classes, num_boxes=self.num_obj)
        if DEBUG2: Save_File(self.Loss_Gradient, "result/loss_gradient")
        if DEBUG2: Save_File(self.Loss, "result/Loss")
        # if DEBUG2:
        #     origin0 = pd.DataFrame(self.Loss_Gradient.view(-1))
        #     # Loss_Grad = np.array(self.Loss_Gradient)
        #     Loss_Grad = self.Loss_Gradient.cpu().detach().numpy()
        #     test_out = 'result/Loss_Grad.txt'
        #     with open(test_out, 'w+') as test_output:
        #         for item in Loss_Grad:
        #             test_output.write(str(item) + "\n")
        #     test_output.close()  
            
                                                    
    def Before_Backward(self,data):
        self.YOLOv2TinyFPGA.Pre_Processing_Backward(self, self.Loss_Gradient)

    def Backward(self,data):
        s = time.time()
        if DEBUG: print("Backward Start")
        self.gWeight, self.gBias, self.gBeta, self.gGamma = self.YOLOv2TinyFPGA.Backward(self,self.Loss_Gradient)
        e = time.time()
        if DEBUG: print("Backward Process Time : ",e-s)
        # self.change_color_red()
        

    