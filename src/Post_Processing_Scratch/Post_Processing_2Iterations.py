from Post_Processing_Scratch.Post_Processing_Function import *
from Post_Processing_Scratch.Calculate_Loss_2Iterations import *
# from Post_Processing_Scratch.Dataset_2Iterations import *
import torch

import pickle
import sys
sys.path.append("../")

def load_pickle():
    default_data = "/home/msis/Desktop/pcie_python/GUI/Post_Processing_Scratch/Input_Data_Batch8.pickle"
    with open(default_data, 'rb') as handle:
        b = pickle.load(handle)
    im_data, gt_boxes, gt_classes, num_boxes = b
    return im_data, gt_boxes, gt_classes, num_boxes


# Conditions for Reading Original_Data from Hardware Processing:
class Post_Processing:
    def __init__(self, Mode, Brain_Floating_Point, Exponent_Bits, Mantissa_Bits, OutImage1_Data_CH0, OutImage1_Data_CH1,
                 OutImage2_Data_CH0, OutImage2_Data_CH1, OutImage3_Data_CH0, OutImage3_Data_CH1, 
                 OutImage4_Data_CH0, OutImage4_Data_CH1, OutImage5_Data_CH0, OutImage5_Data_CH1,
                 OutImage6_Data_CH0, OutImage6_Data_CH1, OutImage7_Data_CH0, OutImage7_Data_CH1,
                 OutImage8_Data_CH0, OutImage8_Data_CH1):
        self.Mode = Mode
        self.Brain_Floating_Point = Brain_Floating_Point
        self.Exponent_Bits = Exponent_Bits
        self.Mantissa_Bits = Mantissa_Bits
        self.OutImage1_Data_CH0 = OutImage1_Data_CH0
        self.OutImage1_Data_CH1 = OutImage1_Data_CH1
        self.OutImage2_Data_CH0 = OutImage2_Data_CH0
        self.OutImage2_Data_CH1 = OutImage2_Data_CH1
        self.OutImage3_Data_CH0 = OutImage3_Data_CH0
        self.OutImage3_Data_CH1 = OutImage3_Data_CH1
        self.OutImage4_Data_CH0 = OutImage4_Data_CH0
        self.OutImage4_Data_CH1 = OutImage4_Data_CH1
        self.OutImage5_Data_CH0 = OutImage5_Data_CH0
        self.OutImage5_Data_CH1 = OutImage5_Data_CH1
        self.OutImage6_Data_CH0 = OutImage6_Data_CH0
        self.OutImage6_Data_CH1 = OutImage6_Data_CH1
        self.OutImage7_Data_CH0 = OutImage7_Data_CH0
        self.OutImage7_Data_CH1 = OutImage7_Data_CH1
        self.OutImage8_Data_CH0 = OutImage8_Data_CH0
        self.OutImage8_Data_CH1 = OutImage8_Data_CH1
        
        
    def PostProcessing(self, gt_boxes, gt_classes, num_boxes):
        global Layer8_Loss_Gradient, Numerical_Loss_

        if self.Brain_Floating_Point:
            Output_Image1 = OutFmap_Layer8_BFPtoDec(self.OutImage1_Data_CH0, self.OutImage1_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image2 = OutFmap_Layer8_BFPtoDec(self.OutImage2_Data_CH0, self.OutImage2_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image3 = OutFmap_Layer8_BFPtoDec(self.OutImage3_Data_CH0, self.OutImage3_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image4 = OutFmap_Layer8_BFPtoDec(self.OutImage4_Data_CH0, self.OutImage4_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image5 = OutFmap_Layer8_BFPtoDec(self.OutImage5_Data_CH0, self.OutImage5_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image6 = OutFmap_Layer8_BFPtoDec(self.OutImage6_Data_CH0, self.OutImage6_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image7 = OutFmap_Layer8_BFPtoDec(self.OutImage7_Data_CH0, self.OutImage7_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image8 = OutFmap_Layer8_BFPtoDec(self.OutImage8_Data_CH0, self.OutImage8_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image = Output_Image1 + Output_Image2 + Output_Image3 + Output_Image4 + \
                           Output_Image5 + Output_Image6 + Output_Image7 + Output_Image8
        else:
            Output_Image1 = OutFmap_Layer8_FP32toDec(self.OutImage1_Data_CH0, self.OutImage1_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image2 = OutFmap_Layer8_FP32toDec(self.OutImage2_Data_CH0, self.OutImage2_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image3 = OutFmap_Layer8_FP32toDec(self.OutImage3_Data_CH0, self.OutImage3_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image4 = OutFmap_Layer8_FP32toDec(self.OutImage4_Data_CH0, self.OutImage4_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image5 = OutFmap_Layer8_FP32toDec(self.OutImage5_Data_CH0, self.OutImage5_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image6 = OutFmap_Layer8_FP32toDec(self.OutImage6_Data_CH0, self.OutImage6_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image7 = OutFmap_Layer8_FP32toDec(self.OutImage7_Data_CH0, self.OutImage7_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image8 = OutFmap_Layer8_FP32toDec(self.OutImage8_Data_CH0, self.OutImage8_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image = Output_Image1 + Output_Image2 + Output_Image3 + Output_Image4 + \
                           Output_Image5 + Output_Image6 + Output_Image7 + Output_Image8

        # Mode is Training
        if self.Mode == "Training":
            # print("\t === Our Post-Processing is finished! ===\n")
            # Convert Output Image into List of Floating 32 Format
            Float_OutputImage = [np.float32(x) for x in Output_Image]
            # print("Float_OutputImage len : ", len(Float_OutputImage))
            test_out = 'Float_OutputImage.txt'
            with open(test_out, 'w+') as test_output:
                for item in Float_OutputImage:
                    line = str(item) 
                    test_output.write(line + '\n')   
            test_output.close()

            # Loss Calculation and Loss Gradient Calculation
            Layer8_Loss, Layer8_Loss_Gradient = loss(Float_OutputImage, gt_boxes=gt_boxes,
                                                     gt_classes=gt_classes, num_boxes=num_boxes)
            Numerical_Loss_ = Numerical_Loss(Layer8_Loss)
            
        # Mode is Inference   
        if self.Mode == "Inference":
            # print("\t === Our Post-Processing is finished! ===\n")
            # Convert Output Image into List of Floating 32 Format
            # Float_OutputImage = [np.float32(x) for x in Output_Image]
            # Loss Calculation and Loss Gradient Calculation
            Layer8_Loss, Layer8_Loss_Gradient = loss(self.OutImage_Data, gt_boxes=None,
                                                     gt_classes=None, num_boxes=None)
            Numerical_Loss_ = Numerical_Loss(Layer8_Loss)
            print("\t === The Backward is starting! ===")

        return Numerical_Loss_, Layer8_Loss_Gradient

