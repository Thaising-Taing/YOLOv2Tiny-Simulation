import sys
sys.path.append("../")

import os
import sys
sys.path.append("../")
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),"Dataset"))
sys.path.append(os.path.join(os.getcwd(),"src"))
sys.path.append(os.path.join(os.getcwd(),"src/Main_Processing_Scratch"))
sys.path.append(os.path.join(os.getcwd(),"src/Pre_Processing_Scratch"))
sys.path.append(os.path.join(os.getcwd(),"src/Post_Processing_Scratch"))
                             
from Pre_Processing_Scratch.Pre_Processing_Function import *
from Pre_Processing_Scratch.WeightLoader import *
import time
import struct
import numpy as np

# from Pre_Processing_Function import *
# Floating Point Parameters that we use
FloatingPoint_Format = "FP32"

# For the BFP16, we don't need to select format, we just use Single Precision, Truncated and Rounding

Selected_FP_Format = {
    "FP32": (8, 23),
    "Bfloat16": (5, 10),
    "Custom": (7, 8)
}

Exponent_Bits, Mantissa_Bits = Selected_FP_Format.get(FloatingPoint_Format, (0, 0))


class Pre_Processing:
    def __init__(self, Mode, Brain_Floating_Point, Exponent_Bits, Mantissa_Bits):
        self.Mode = Mode
        self.Brain_Floating_Point = Brain_Floating_Point
        self.Exponent_Bits = Exponent_Bits
        self.Mantissa_Bits = Mantissa_Bits

        # Image Width and Height
        self.frameWidth = 416
        self.frameHeight = 416

    def Image_Converted_Func(self, Image):
        # Mode is Training
        # global Image
        if self.Mode == "Training":
            # # Loading Image & Write the Image
            # print("\t --> " + " The Image is selecting and Converting!")
            
            if self.Brain_Floating_Point:
                image1 = Read_Image_into_BFP(Image[0:1].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image2 = Read_Image_into_BFP(Image[1:2].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image3 = Read_Image_into_BFP(Image[2:3].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image4 = Read_Image_into_BFP(Image[3:4].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image5 = Read_Image_into_BFP(Image[4:5].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image6 = Read_Image_into_BFP(Image[5:6].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image7 = Read_Image_into_BFP(Image[6:7].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image8 = Read_Image_into_BFP(Image[7:8].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                Total_Image = [image1, image2, image3, image4, image5, image6, image7, image8]
            else:
                # Image = Read_Image_into_FP32(Image, self.Exponent_Bits, self.Mantissa_Bits)
                image1 = Read_Image_into_FP32(Image[0:1].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image2 = Read_Image_into_FP32(Image[1:2].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image3 = Read_Image_into_FP32(Image[2:3].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image4 = Read_Image_into_FP32(Image[3:4].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image5 = Read_Image_into_FP32(Image[4:5].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image6 = Read_Image_into_FP32(Image[5:6].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image7 = Read_Image_into_FP32(Image[6:7].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image8 = Read_Image_into_FP32(Image[7:8].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                Total_Image = [image1, image2, image3, image4, image5, image6, image7, image8]
        # Mode is Inference
        if self.Mode == "Inference":
            if self.Brain_Floating_Point:
                image1 = Read_Image_into_BFP(Image[0:1].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image2 = Read_Image_into_BFP(Image[1:2].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image3 = Read_Image_into_BFP(Image[2:3].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image4 = Read_Image_into_BFP(Image[3:4].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image5 = Read_Image_into_BFP(Image[4:5].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image6 = Read_Image_into_BFP(Image[5:6].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image7 = Read_Image_into_BFP(Image[6:7].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image8 = Read_Image_into_BFP(Image[7:8].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                Total_Image = [image1, image2, image3, image4, image5, image6, image7, image8]
            else:
                # Image = Read_Image_into_FP32(Image, self.Exponent_Bits, self.Mantissa_Bits)
                image1 = Read_Image_into_FP32(Image[0:1].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image2 = Read_Image_into_FP32(Image[1:2].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image3 = Read_Image_into_FP32(Image[2:3].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image4 = Read_Image_into_FP32(Image[3:4].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image5 = Read_Image_into_FP32(Image[4:5].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image6 = Read_Image_into_FP32(Image[5:6].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image7 = Read_Image_into_FP32(Image[6:7].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image8 = Read_Image_into_FP32(Image[7:8].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                Total_Image = [image1, image2, image3, image4, image5, image6, image7, image8]
        return Total_Image



    def Image_Converted_Func_Test(self, Image):
        # Mode is Training
        # global Image
        if self.Mode == "Training":
            # # Loading Image & Write the Image
            print("\t --> " + " The Image is selecting and Converting!")
            
            if self.Brain_Floating_Point:
                image1 = Read_Image_into_BFP(Image[0:1].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image2 = Read_Image_into_BFP(Image[1:2].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image3 = Read_Image_into_BFP(Image[2:3].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image4 = Read_Image_into_BFP(Image[3:4].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image5 = Read_Image_into_BFP(Image[4:5].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image6 = Read_Image_into_BFP(Image[5:6].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image7 = Read_Image_into_BFP(Image[6:7].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image8 = Read_Image_into_BFP(Image[7:8].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                Total_Image = [image1, image2, image3, image4, image5, image6, image7, image8]
            else:
                # Image = Read_Image_into_FP32(Image, self.Exponent_Bits, self.Mantissa_Bits)
                image1 = Read_Image_into_FP32(Image[0:1].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image2 = Read_Image_into_FP32(Image[1:2].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image3 = Read_Image_into_FP32(Image[2:3].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image4 = Read_Image_into_FP32(Image[3:4].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image5 = Read_Image_into_FP32(Image[4:5].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image6 = Read_Image_into_FP32(Image[5:6].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image7 = Read_Image_into_FP32(Image[6:7].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                image8 = Read_Image_into_FP32(Image[7:8].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                Total_Image = [image1, image2, image3, image4, image5, image6, image7, image8]
        # Mode is Inference
        if self.Mode == "Inference":
            if self.Brain_Floating_Point:
                image1 = Read_Image_into_BFP(Image[0:1].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                Total_Image = [image1]
            else:
                # Image = Read_Image_into_FP32(Image, self.Exponent_Bits, self.Mantissa_Bits)
                image1 = Read_Image_into_FP32(Image[0:1].flatten().tolist(), self.Exponent_Bits, self.Mantissa_Bits)
                Total_Image = [image1]
        return Total_Image


    def WeightLoader(self):
        global Weight, Bias, Beta, Gamma, Running_Mean, Running_Var
        if self.Mode == "Training":
            pytorch_model = DeepConvNet(input_dims=(3, 416, 416),
                                        num_filters=[16, 32, 64, 128, 256, 512, 1024, 1024],
                                        max_pools=[0, 1, 2, 3, 4],
                                        weight_scale='kaiming',
                                        batchnorm=True,
                                        dtype=torch.float32, device='cpu')

            model = Yolov2()
            weightloader = WeightLoader()
            Data_Path = "Dataset/Dataset/pretrained/yolov2-tiny-voc.weights"
            pytorch_model = weightloader.load(pytorch_model, model, Data_Path)
            Weight, Bias, Beta, Gamma, Running_Mean, Running_Var = pytorch_model.Training_Parameters()
        return Weight, Bias, Gamma, Beta, Running_Mean, Running_Var

    def Weight_Converted_Func(self, Weight_Dec, Bias_Dec, Beta_Dec, Gamma_Dec, Running_Mean_Dec, Running_Var_Dec):
        # Mode is Training
        if self.Mode == "Training":
            # Weight
            Weight0 = Read_Weight_into_Bfloat16(Weight_Dec[0], self.Exponent_Bits, self.Mantissa_Bits)
            Weight1 = Read_Weight_into_Bfloat16(Weight_Dec[1], self.Exponent_Bits, self.Mantissa_Bits)
            Weight2 = Read_Weight_into_Bfloat16(Weight_Dec[2], self.Exponent_Bits, self.Mantissa_Bits)
            Weight3 = Read_Weight_into_Bfloat16(Weight_Dec[3], self.Exponent_Bits, self.Mantissa_Bits)
            Weight4 = Read_Weight_into_Bfloat16(Weight_Dec[4], self.Exponent_Bits, self.Mantissa_Bits)
            Weight5 = Read_Weight_into_Bfloat16(Weight_Dec[5], self.Exponent_Bits, self.Mantissa_Bits)
            Weight6 = Read_Weight_into_Bfloat16(Weight_Dec[6], self.Exponent_Bits, self.Mantissa_Bits)
            Weight7 = Read_Weight_into_Bfloat16(Weight_Dec[7], self.Exponent_Bits, self.Mantissa_Bits)
            Weight8 = Read_Weight_into_Bfloat16(Weight_Dec[8], self.Exponent_Bits, self.Mantissa_Bits)
            Weight_Bfloat = [Weight0, Weight1, Weight2, Weight3, Weight4, Weight5, Weight6, Weight7, Weight8]
            # Bias 
            Bias_Bfloat = Read_Weight_into_Bfloat16(Bias_Dec, self.Exponent_Bits, self.Mantissa_Bits)
            # Beta
            Beta0 = Read_Weight_into_Bfloat16(Beta_Dec[0], self.Exponent_Bits, self.Mantissa_Bits)
            Beta1 = Read_Weight_into_Bfloat16(Beta_Dec[1], self.Exponent_Bits, self.Mantissa_Bits)
            Beta2 = Read_Weight_into_Bfloat16(Beta_Dec[2], self.Exponent_Bits, self.Mantissa_Bits)
            Beta3 = Read_Weight_into_Bfloat16(Beta_Dec[3], self.Exponent_Bits, self.Mantissa_Bits)
            Beta4 = Read_Weight_into_Bfloat16(Beta_Dec[4], self.Exponent_Bits, self.Mantissa_Bits)
            Beta5 = Read_Weight_into_Bfloat16(Beta_Dec[5], self.Exponent_Bits, self.Mantissa_Bits)
            Beta6 = Read_Weight_into_Bfloat16(Beta_Dec[6], self.Exponent_Bits, self.Mantissa_Bits)
            Beta7 = Read_Weight_into_Bfloat16(Beta_Dec[7], self.Exponent_Bits, self.Mantissa_Bits)
            Beta_Bfloat = [Beta0, Beta1, Beta2, Beta3, Beta4, Beta5, Beta6, Beta7]
            # Gamma
            Gamma0 = Read_Weight_into_Bfloat16(Gamma_Dec[0], self.Exponent_Bits, self.Mantissa_Bits)
            Gamma1 = Read_Weight_into_Bfloat16(Gamma_Dec[1], self.Exponent_Bits, self.Mantissa_Bits)
            Gamma2 = Read_Weight_into_Bfloat16(Gamma_Dec[2], self.Exponent_Bits, self.Mantissa_Bits)
            Gamma3 = Read_Weight_into_Bfloat16(Gamma_Dec[3], self.Exponent_Bits, self.Mantissa_Bits)
            Gamma4 = Read_Weight_into_Bfloat16(Gamma_Dec[4], self.Exponent_Bits, self.Mantissa_Bits)
            Gamma5 = Read_Weight_into_Bfloat16(Gamma_Dec[5], self.Exponent_Bits, self.Mantissa_Bits)
            Gamma6 = Read_Weight_into_Bfloat16(Gamma_Dec[6], self.Exponent_Bits, self.Mantissa_Bits)
            Gamma7 = Read_Weight_into_Bfloat16(Gamma_Dec[7], self.Exponent_Bits, self.Mantissa_Bits)
            Gamma_Bfloat = [Gamma0, Gamma1, Gamma2, Gamma3, Gamma4, Gamma5, Gamma6, Gamma7]
            # Running Mean
            Running_Mean0 = Read_Weight_into_Bfloat16(Running_Mean_Dec[0], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Mean1 = Read_Weight_into_Bfloat16(Running_Mean_Dec[1], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Mean2 = Read_Weight_into_Bfloat16(Running_Mean_Dec[2], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Mean3 = Read_Weight_into_Bfloat16(Running_Mean_Dec[3], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Mean4 = Read_Weight_into_Bfloat16(Running_Mean_Dec[4], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Mean5 = Read_Weight_into_Bfloat16(Running_Mean_Dec[5], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Mean6 = Read_Weight_into_Bfloat16(Running_Mean_Dec[6], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Mean7 = Read_Weight_into_Bfloat16(Running_Mean_Dec[7], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Mean_Bfloat = [Running_Mean0, Running_Mean1, Running_Mean2, Running_Mean3, Running_Mean4, Running_Mean5, Running_Mean6, Running_Mean7]
            # Running Var
            Running_Var0 = Read_Weight_into_Bfloat16(Running_Var_Dec[0], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Var1 = Read_Weight_into_Bfloat16(Running_Var_Dec[1], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Var2 = Read_Weight_into_Bfloat16(Running_Var_Dec[2], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Var3 = Read_Weight_into_Bfloat16(Running_Var_Dec[3], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Var4 = Read_Weight_into_Bfloat16(Running_Var_Dec[4], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Var5 = Read_Weight_into_Bfloat16(Running_Var_Dec[5], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Var6 = Read_Weight_into_Bfloat16(Running_Var_Dec[6], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Var7 = Read_Weight_into_Bfloat16(Running_Var_Dec[7], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Var_Bfloat = [Running_Var0, Running_Var1, Running_Var2, Running_Var3, Running_Var4, Running_Var5, Running_Var6, Running_Var7]

        else:
            Weight0 = Read_Weight_into_Bfloat16(Weight_Dec[0], self.Exponent_Bits, self.Mantissa_Bits)
            Weight1 = Read_Weight_into_Bfloat16(Weight_Dec[1], self.Exponent_Bits, self.Mantissa_Bits)
            Weight2 = Read_Weight_into_Bfloat16(Weight_Dec[2], self.Exponent_Bits, self.Mantissa_Bits)
            Weight3 = Read_Weight_into_Bfloat16(Weight_Dec[3], self.Exponent_Bits, self.Mantissa_Bits)
            Weight4 = Read_Weight_into_Bfloat16(Weight_Dec[4], self.Exponent_Bits, self.Mantissa_Bits)
            Weight5 = Read_Weight_into_Bfloat16(Weight_Dec[5], self.Exponent_Bits, self.Mantissa_Bits)
            Weight6 = Read_Weight_into_Bfloat16(Weight_Dec[6], self.Exponent_Bits, self.Mantissa_Bits)
            Weight7 = Read_Weight_into_Bfloat16(Weight_Dec[7], self.Exponent_Bits, self.Mantissa_Bits)
            Weight8 = Read_Weight_into_Bfloat16(Weight_Dec[8], self.Exponent_Bits, self.Mantissa_Bits)
            Weight_Bfloat = [Weight0, Weight1, Weight2, Weight3, Weight4, Weight5, Weight6, Weight7, Weight8]
            # Bias 
            Bias_Bfloat = Read_Weight_into_Bfloat16(Bias_Dec, self.Exponent_Bits, self.Mantissa_Bits)
            # Beta
            Beta0 = Read_Weight_into_Bfloat16(Beta_Dec[0], self.Exponent_Bits, self.Mantissa_Bits)
            Beta1 = Read_Weight_into_Bfloat16(Beta_Dec[1], self.Exponent_Bits, self.Mantissa_Bits)
            Beta2 = Read_Weight_into_Bfloat16(Beta_Dec[2], self.Exponent_Bits, self.Mantissa_Bits)
            Beta3 = Read_Weight_into_Bfloat16(Beta_Dec[3], self.Exponent_Bits, self.Mantissa_Bits)
            Beta4 = Read_Weight_into_Bfloat16(Beta_Dec[4], self.Exponent_Bits, self.Mantissa_Bits)
            Beta5 = Read_Weight_into_Bfloat16(Beta_Dec[5], self.Exponent_Bits, self.Mantissa_Bits)
            Beta6 = Read_Weight_into_Bfloat16(Beta_Dec[6], self.Exponent_Bits, self.Mantissa_Bits)
            Beta7 = Read_Weight_into_Bfloat16(Beta_Dec[7], self.Exponent_Bits, self.Mantissa_Bits)
            Beta_Bfloat = [Beta0, Beta1, Beta2, Beta3, Beta4, Beta5, Beta6, Beta7]
            # Gamma
            Gamma0 = Read_Weight_into_Bfloat16(Gamma_Dec[0], self.Exponent_Bits, self.Mantissa_Bits)
            Gamma1 = Read_Weight_into_Bfloat16(Gamma_Dec[1], self.Exponent_Bits, self.Mantissa_Bits)
            Gamma2 = Read_Weight_into_Bfloat16(Gamma_Dec[2], self.Exponent_Bits, self.Mantissa_Bits)
            Gamma3 = Read_Weight_into_Bfloat16(Gamma_Dec[3], self.Exponent_Bits, self.Mantissa_Bits)
            Gamma4 = Read_Weight_into_Bfloat16(Gamma_Dec[4], self.Exponent_Bits, self.Mantissa_Bits)
            Gamma5 = Read_Weight_into_Bfloat16(Gamma_Dec[5], self.Exponent_Bits, self.Mantissa_Bits)
            Gamma6 = Read_Weight_into_Bfloat16(Gamma_Dec[6], self.Exponent_Bits, self.Mantissa_Bits)
            Gamma7 = Read_Weight_into_Bfloat16(Gamma_Dec[7], self.Exponent_Bits, self.Mantissa_Bits)
            Gamma_Bfloat = [Gamma0, Gamma1, Gamma2, Gamma3, Gamma4, Gamma5, Gamma6, Gamma7]
            # Running Mean
            Running_Mean0 = Read_Weight_into_Bfloat16(Running_Mean_Dec[0], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Mean1 = Read_Weight_into_Bfloat16(Running_Mean_Dec[1], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Mean2 = Read_Weight_into_Bfloat16(Running_Mean_Dec[2], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Mean3 = Read_Weight_into_Bfloat16(Running_Mean_Dec[3], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Mean4 = Read_Weight_into_Bfloat16(Running_Mean_Dec[4], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Mean5 = Read_Weight_into_Bfloat16(Running_Mean_Dec[5], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Mean6 = Read_Weight_into_Bfloat16(Running_Mean_Dec[6], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Mean7 = Read_Weight_into_Bfloat16(Running_Mean_Dec[7], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Mean_Bfloat = [Running_Mean0, Running_Mean1, Running_Mean2, Running_Mean3, Running_Mean4, Running_Mean5, Running_Mean6, Running_Mean7]
            # Running Var
            Running_Var0 = Read_Weight_into_Bfloat16(Running_Var_Dec[0], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Var1 = Read_Weight_into_Bfloat16(Running_Var_Dec[1], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Var2 = Read_Weight_into_Bfloat16(Running_Var_Dec[2], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Var3 = Read_Weight_into_Bfloat16(Running_Var_Dec[3], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Var4 = Read_Weight_into_Bfloat16(Running_Var_Dec[4], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Var5 = Read_Weight_into_Bfloat16(Running_Var_Dec[5], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Var6 = Read_Weight_into_Bfloat16(Running_Var_Dec[6], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Var7 = Read_Weight_into_Bfloat16(Running_Var_Dec[7], self.Exponent_Bits, self.Mantissa_Bits)
            Running_Var_Bfloat = [Running_Var0, Running_Var1, Running_Var2, Running_Var3, Running_Var4, Running_Var5, Running_Var6, Running_Var7]    
            
        return Weight_Bfloat, Bias_Bfloat, Beta_Bfloat, Gamma_Bfloat, Running_Mean_Bfloat, Running_Var_Bfloat

    def Bias_Converted_Func(self):
        # Mode is Training
        global Bias, Bias_Dec
        if self.Mode == "Training":
            # Loading Bias and Write Bias:
            Read_Bias_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Bias"
            File_List_Bias = os.listdir(Read_Bias_Folder_Path)
            if self.Brain_Floating_Point:
                Bias_Dec = Read_Bias(File_List_Bias, Read_Bias_Folder_Path)
                Bias = Bias_Bfloat16(Bias_Dec, self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Bias_Dec = Read_Bias(File_List_Bias, Read_Bias_Folder_Path)
                Bias = Bias_FP32(Bias_Dec, self.Exponent_Bits, self.Mantissa_Bits)

        # Mode is Inference
        if self.Mode == "Inference":
            # Loading Bias and Write Bias:
            Read_Bias_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Bias"
            File_List_Bias = os.listdir(Read_Bias_Folder_Path)
            if self.Brain_Floating_Point:
                Bias_Dec = Read_Bias(File_List_Bias, Read_Bias_Folder_Path,
                                          self.Exponent_Bits, self.Mantissa_Bits)
                Bias = Bias_Bfloat16(Bias_Dec, self.Exponent_Bits, self.Mantissa_Bits)
            else: 
                Bias_Dec = Read_Bias(File_List_Bias, Read_Bias_Folder_Path)
                Bias = Bias_FP32(Bias_Dec, self.Exponent_Bits, self.Mantissa_Bits)
                
        return Bias_Dec, Bias

    def Beta_Param_Converted_Func(self):
        # Mode is Training
        global Beta_List
        if self.Mode == "Training":
            # Loading Beta and Write Beta:
            Read_Beta_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Beta"
            File_List_Beta = os.listdir(Read_Beta_Folder_Path)
            if self.Brain_Floating_Point:
                Beta_Dec_List = Read_BN(File_List_Beta, Read_Beta_Folder_Path)
                Beta_List = BN_Bfloat16(Beta_Dec_List, self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Beta_Dec_List = Read_BN(File_List_Beta, Read_Beta_Folder_Path)
                Beta_List = BN_FP32(Beta_Dec_List, self.Exponent_Bits, self.Mantissa_Bits)

        # Mode is Inference
        if self.Mode == "Inference":
            # Loading Beta and Write Beta:
            Read_Beta_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Beta"
            File_List_Beta = os.listdir(Read_Beta_Folder_Path)
            if self.Brain_Floating_Point:
                Beta_Dec_List = Read_BN(File_List_Beta, Read_Beta_Folder_Path)
                Beta_List = BN_Bfloat16(Beta_Dec_List, self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Beta_Dec_List = Read_BN(File_List_Beta, Read_Beta_Folder_Path)
                Beta_List = BN_FP32(Beta_Dec_List, self.Exponent_Bits, self.Mantissa_Bits)
                
        return Beta_Dec_List, Beta_List

    def Gamma_Param_Converted_Func(self):
        # Mode is Training
        global Gamma_List
        if self.Mode == "Training":
            # Loading Gamma and Write Gamma:
            Read_Gamma_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Gamma"
            File_List_Gamma = os.listdir(Read_Gamma_Folder_Path)
            if self.Brain_Floating_Point:
                Gamma_Dec_List = Read_BN(File_List_Gamma, Read_Gamma_Folder_Path)
                Gamma_List = BN_Bfloat16(Gamma_Dec_List, self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Gamma_Dec_List = Read_BN(File_List_Gamma, Read_Gamma_Folder_Path)
                Gamma_List = BN_FP32(Gamma_Dec_List, self.Exponent_Bits, self.Mantissa_Bits)

        # Mode is Inference
        if self.Mode == "Inference":
            # Loading Gamma and Write Gamma:
            Read_Gamma_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Gamma"
            File_List_Gamma = os.listdir(Read_Gamma_Folder_Path)
            if self.Brain_Floating_Point:
                Gamma_Dec_List = Read_BN(File_List_Gamma, Read_Gamma_Folder_Path)
                Gamma_List = BN_Bfloat16(Gamma_Dec_List, self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Gamma_Dec_List = Read_BN(File_List_Gamma, Read_Gamma_Folder_Path)
                Gamma_List = BN_FP32(Gamma_Dec_List, self.Exponent_Bits, self.Mantissa_Bits)

        return Gamma_Dec_List, Gamma_List

    def Running_Mean_Param_Converted_Func(self):
        # Mode is Training
        global Running_Mean_List
        if self.Mode == "Training":
            # Loading Running_Mean and Write Running_Mean:
            Read_RunningMean_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Running_Mean"
            File_List_RunningMean = os.listdir(Read_RunningMean_Folder_Path)
            if self.Brain_Floating_Point:
                Running_Mean_Dec_List = Read_BN(File_List_RunningMean, Read_RunningMean_Folder_Path)
                Running_Mean_List = BN_Bfloat16(Running_Mean_Dec_List, self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Running_Mean_Dec_List = Read_BN(File_List_RunningMean, Read_RunningMean_Folder_Path)
                Running_Mean_List = BN_FP32(Running_Mean_Dec_List, self.Exponent_Bits, self.Mantissa_Bits)
                
        if self.Mode == "Inference":
            # Loading Running_Mean and Write Running_Mean:
            Read_RunningMean_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Running_Mean"
            File_List_RunningMean = os.listdir(Read_RunningMean_Folder_Path)
            if self.Brain_Floating_Point:
                Running_Mean_Dec_List = Read_BN(File_List_RunningMean, Read_RunningMean_Folder_Path)
                Running_Mean_List = BN_Bfloat16(Running_Mean_Dec_List, self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Running_Mean_Dec_List = Read_BN(File_List_RunningMean, Read_RunningMean_Folder_Path)
                Running_Mean_List = BN_FP32(Running_Mean_Dec_List, self.Exponent_Bits, self.Mantissa_Bits)

        return Running_Mean_Dec_List, Running_Mean_List

    def Running_Var_Param_Converted_Func(self):
        # Mode is Training
        global Running_Var_List
        if self.Mode == "Training":
            # Loading Running_Var and Write Running_Var:
            Read_RunningVar_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Running_Var"
            File_List_RunningVar = os.listdir(Read_RunningVar_Folder_Path)
            if self.Brain_Floating_Point:
                Running_Var_Dec_List = Read_BN(File_List_RunningVar, Read_RunningVar_Folder_Path)
                Running_Var_List = BN_Bfloat16(Running_Var_Dec_List, self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Running_Var_Dec_List = Read_BN(File_List_RunningVar, Read_RunningVar_Folder_Path)
                Running_Var_List = BN_FP32(Running_Var_Dec_List, self.Exponent_Bits, self.Mantissa_Bits)

        # Mode is Inference
        if self.Mode == "Inference":
            # Loading Running_Var and Write Running_Var:
            Read_RunningVar_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Running_Var"
            File_List_RunningVar = os.listdir(Read_RunningVar_Folder_Path)
            if self.Brain_Floating_Point:
                Running_Var_Dec_List = Read_BN(File_List_RunningVar, Read_RunningVar_Folder_Path)
                Running_Var_List = BN_Bfloat16(Running_Var_Dec_List, self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Running_Var_Dec_List = Read_BN(File_List_RunningVar, Read_RunningVar_Folder_Path)
                Running_Var_List = BN_FP32(Running_Var_Dec_List, self.Exponent_Bits, self.Mantissa_Bits)

        return Running_Var_Dec_List, Running_Var_List
    
    def Add_Param_Converted_Func(self):
        # Mode is Training
        global Add_List
        if self.Mode == "Training":
            # Loading Running_Var and Write Running_Var:
            print("\t --> " + " The Add Param: Add_Params are processing, Please Wait for a Moment!")
            Read_Add_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Add"
            File_List_Add = os.listdir(Read_Add_Folder_Path)
            if self.Brain_Floating_Point:
                Add_List = Read_BN_Parameters_into_BFP(File_List_Add, Read_Add_Folder_Path,
                                            self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Add_List = Read_BN_Parameters_into_FP32(File_List_Add, Read_Add_Folder_Path,
                                             self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The Add Param: Add_Params are Successfully Converted!")

        # Mode is Inference
        if self.Mode == "Inference":
            # Loading Running_Var and Write Running_Var:
            print("\t --> " + " The Add Param: Add_Params are processing, Please Wait for a Moment!")
            Read_Add_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Add"
            File_List_Add = os.listdir(Read_Add_Folder_Path)
            if self.Brain_Floating_Point:
                Add_List = Read_BN_Parameters_into_BFP(File_List_Add, Read_Add_Folder_Path,
                                                               self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Add_List = Read_BN_Parameters_into_FP32(File_List_Add, Read_Add_Folder_Path,
                                                                self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The Add Param: Add_Params are Successfully Converted!")

        return Add_List
    
    def Mul_Param_Converted_Func(self):
        # Mode is Training
        global Mul_List
        if self.Mode == "Training":
            # Loading Running_Var and Write Running_Var:
            print("\t --> " + " The Mul Param: Mul_Params are processing, Please Wait for a Moment!")
            Read_Mul_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Mul"
            File_List_Mul = os.listdir(Read_Mul_Folder_Path)
            if self.Brain_Floating_Point:
                Mul_List = Read_BN_Parameters_into_BFP(File_List_Mul, Read_Mul_Folder_Path,
                                            self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Mul_List = Read_BN_Parameters_into_FP32(File_List_Mul, Read_Mul_Folder_Path,
                                             self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The Mul Param: Mul_Params are Successfully Converted!")

        # Mode is Inference
        if self.Mode == "Inference":
            # Loading Running_Var and Write Running_Var:
            print("\t --> " + " The Mul Param: Mul_Params are processing, Please Wait for a Moment!")
            Read_Mul_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Mul"
            File_List_Mul = os.listdir(Read_Mul_Folder_Path)
            if self.Brain_Floating_Point:
                Mul_List = Read_BN_Parameters_into_BFP(File_List_Mul, Read_Mul_Folder_Path,
                                                               self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Mul_List = Read_BN_Parameters_into_FP32(File_List_Mul, Read_Mul_Folder_Path,
                                                                self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The Mul Param: Mul_Params are Successfully Converted!")

        return Mul_List
    
    def Sub_Param_Converted_Func(self):
        # Mode is Training
        global Sub_List
        if self.Mode == "Training":
            # Loading Running_Var and Write Running_Var:
            print("\t --> " + " The Sub Param: Sub_Params are processing, Please Wait for a Moment!")
            Read_Sub_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Sub"
            File_List_Sub = os.listdir(Read_Sub_Folder_Path)
            if self.Brain_Floating_Point:
                Sub_List = Read_BN_Parameters_into_BFP(File_List_Sub, Read_Sub_Folder_Path,
                                            self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Sub_List = Read_BN_Parameters_into_FP32(File_List_Sub, Read_Sub_Folder_Path,
                                             self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The Sub Param: Sub_Params are Successfully Converted!")

        # Mode is Inference
        if self.Mode == "Inference":
            # Loading Running_Var and Write Running_Var:
            print("\t --> " + " The Sub Param: Sub_Params are processing, Please Wait for a Moment!")
            Read_Sub_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Sub"
            File_List_Sub = os.listdir(Read_Sub_Folder_Path)
            if self.Brain_Floating_Point:
                Sub_List = Read_BN_Parameters_into_BFP(File_List_Sub, Read_Sub_Folder_Path,
                                                               self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Sub_List = Read_BN_Parameters_into_FP32(File_List_Sub, Read_Sub_Folder_Path,
                                                                self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The Sub Param: Sub_Params are Successfully Converted!")

        return Sub_List
    
    def Backward_Const_Param_Converted_Func(self):
        # Mode is Training
        global Backward_Const_List
        if self.Mode == "Training":
            # Loading Beta and Write Beta:
            print("\t --> " + " The BN Param: Backward_Constant is processing, Please Wait for a Moment!")
            Read_Backward_Const_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Backward_Const"
            File_List_Backward_Const = os.listdir(Read_Backward_Const_Folder_Path)
            if self.Brain_Floating_Point:
                Backward_Const_List = Read_BN_Parameters_into_BFP(File_List_Backward_Const, Read_Backward_Const_Folder_Path,
                                            self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Backward_Const_List = Read_BN_Parameters_into_FP32(File_List_Backward_Const, Read_Backward_Const_Folder_Path,
                                             self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The BN Param: Backward_Constant is Successfully Converted!")

        # Mode is Inference
        if self.Mode == "Inference":
            # Loading Beta and Write Beta:
            print("\t --> " + " The BN Param: Backward_Constant is processing, Please Wait for a Moment!")
            Read_Backward_Const_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Backward_Const"
            File_List_Backward_Const = os.listdir(Read_Backward_Const_Folder_Path)
            if self.Brain_Floating_Point:
                Backward_Const_List = Read_BN_Parameters_into_BFP(File_List_Backward_Const, Read_Backward_Const_Folder_Path,
                                                        self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Backward_Const_List = Read_BN_Parameters_into_FP32(File_List_Backward_Const, Read_Backward_Const_Folder_Path,
                                                         self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The BN Param: Backward_Constant is Successfully Converted!")
        return Backward_Const_List
    
    def Average_Per_Channel_Param_Converted_Func(self):
        # Mode is Training
        global Average_List
        if self.Mode == "Training":
            # Loading Beta and Write Beta:
            print("\t --> " + " The BN Param: Average_Per_Channel is processing, Please Wait for a Moment!")
            Read_Average_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Average_Per_Channel"
            File_List_Average = os.listdir(Read_Average_Folder_Path)
            if self.Brain_Floating_Point:
                Average_List = Read_BN_Parameters_into_BFP(File_List_Average, Read_Average_Folder_Path,
                                            self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Average_List = Read_BN_Parameters_into_FP32(File_List_Average, Read_Average_Folder_Path,
                                             self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The BN Param: Average_Per_Channel is Successfully Converted!")

        # Mode is Inference
        if self.Mode == "Inference":
            # Loading Beta and Write Beta:
            print("\t --> " + " The BN Param: Average_Per_Channel is processing, Please Wait for a Moment!")
            Read_Average_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Average_Per_Channel"
            File_List_Average = os.listdir(Read_Average_Folder_Path)
            if self.Brain_Floating_Point:
                Average_List = Read_BN_Parameters_into_BFP(File_List_Average, Read_Average_Folder_Path,
                                                        self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Average_List = Read_BN_Parameters_into_FP32(File_List_Average, Read_Average_Folder_Path,
                                                         self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The BN Param: Average_Per_Channel is Successfully Converted!")
        return Average_List
    
'''
def Read_OutFmap_Bfloat2Dec(Data_List_CH0, Data_List_CH1, Exponent_Bit, Mantissa_Bit, Out_CH, Out_Size, Layer8=False):
    # Layer 8: Out_CH=128, Out_Size=13
    # Read all the Weights from a Weight_Folder
    Input_List0 = Data_List_CH0
    Input_List1 = Data_List_CH1
    reverse_time_s = time.time()
    Input_List = Fmap_Reverse_Ordering(Out_CH, Out_Size, Input_List0, Input_List1)
    reverse_time_e = time.time()
    print("odering time : ", reverse_time_e - reverse_time_s)
    Out_List = []
    Out_List.clear()
    if Layer8:
        b2f_time_s = time.time()
        for Value in Input_List[:(len(Input_List)-3*13*13)]:
            Hexadecimal = str(Value) + "0000"
            Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
            Decimal = Binary2Floating(Binary_Value, Exponent_Bit, Mantissa_Bit)
            Out_List.append(str(Decimal))
        b2f_time_e = time.time()
        print("b2f time : ", b2f_time_e - b2f_time_s)    
    else: 
        b2f_time_s = time.time()
        for Value in Input_List[:(len(Input_List))]:
            Hexadecimal = str(Value) + "0000"
            Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
            Decimal = Binary2Floating(Binary_Value, Exponent_Bit, Mantissa_Bit)
            Out_List.append(str(Decimal))   
        b2f_time_e = time.time()
        print("b2f time : ", b2f_time_e - b2f_time_s)            
    return Out_List
'''

def Read_OutFmap_Bfloat2Dec(Data_List_CH0, Data_List_CH1, Exponent_Bit, Mantissa_Bit, Out_CH, Out_Size, Layer8=False):
    # Layer 8: Out_CH=128, Out_Size=13
    # Read all the Weights from a Weight_Folder
    Input_List0 = Data_List_CH0
    Input_List1 = Data_List_CH1
    Input_List = Fmap_Reverse_Ordering(Out_CH, Out_Size, Input_List0, Input_List1)
    Out_List = []
    Out_List.clear()
    if Layer8:
        Out_List = Input_List[:(len(Input_List)-3*13*13)] 
    else: 
        Out_List = Input_List       
    return Out_List

def Read_ReLu_Marking(Data_List_CH0, Data_List_CH1, Exponent_Bit, Mantissa_Bit, Out_CH, Out_Size, Layer8=False):
    # Layer 8: Out_CH=128, Out_Size=13
    # Read all the Weights from a Weight_Folder
    Input_List0 = Data_List_CH0
    Input_List1 = Data_List_CH1
    Input_List = ReLu_Marking_Ordering(Out_CH, Out_Size, Input_List0, Input_List1)
    Out_List = []
    Out_List.clear()
    if Layer8:
        Out_List = Input_List[:(len(Input_List)-3*13*13)] 
    else: 
        Out_List = Input_List       
    return Out_List

def origin_idx_calculator(idx, B, H, W, num_chunks):
    origin_idx = []
    origin_idx.clear()
    if num_chunks < H*W//num_chunks:
        for i in range(len(idx)):
            for j in range(len(idx[0])):
                origin_idx.append([(j*num_chunks*B+int(idx[i][j]))//(H*W), i, 
                        ((j*num_chunks*B+int(idx[i][j]))%(H*W))//H, ((j*num_chunks*B+int(idx[i][j]))%(H*W))%H])
    else:
        for i in range(len(idx)):
            for j in range(len(idx[0])):
                origin_idx.append([(j*B*H*W//num_chunks+int(idx[i][j]))//(H*W), i,
                        ((j*B*H*W//num_chunks+int(idx[i][j]))%(H*W))//H, ((j*B*H*W//num_chunks+int(idx[i][j]))%(H*W))%H])
    return origin_idx


'''
class Cal_mean_var(object):

    @staticmethod
    def forward(x):
    
        out, cache = None, None
        
        eps = 1e-5
        num_chunks = 8
        B, C, H, W = x.shape
        y = x.transpose(0, 1).contiguous()  # C x B x H x W
        y = y.view(C, num_chunks, B * H * W // num_chunks)
        avg_max = y.max(-1)[0].mean(-1)  # C
        avg_min = y.min(-1)[0].mean(-1)  # C
        avg = y.view(C, -1).mean(-1)  # C
        max_index = origin_idx_calculator(y.max(-1)[1], B, H, W, num_chunks)
        min_index = origin_idx_calculator(y.min(-1)[1], B, H, W, num_chunks)
        scale_fix = 1 / ((2 * math.log(y.size(-1))) ** 0.5)
        # scale_fix = 1 / ((2 * math.log(num_chunks)) ** 0.5)
        scale = 1 / ((avg_max - avg_min) * scale_fix + eps) 
        # scale = 1 / ((avg_max - avg_min) * scale_fix) 

        avg = avg.view(1, -1, 1, 1)
        scale = scale.view(1, -1, 1, 1)

        cache = x
        return avg, scale
    
    @staticmethod
    def backward(dout, cache):
        
        x = cache
        B, C, H, W = x.shape
        dL_davg = (dout).sum(dim=(0, 2, 3), keepdim=True)
        avg_pc = dL_davg / (B * H * W)
        
        
        return avg_pc
'''

class Cal_mean_var(object):
    import math
    @staticmethod
    def forward(x):
        out, cache = None, None
        eps = 1e-5
        num_chunks = 8
        B, C, H, W = x.shape
        y = x.transpose(0, 1).contiguous()  # C x B x H x W
        y = y.view(C, num_chunks, B * H * W // num_chunks)
        avg_max = y.max(-1)[0].mean(-1)  # C
        avg_min = y.min(-1)[0].mean(-1)  # C
        avg = y.view(C, -1).mean(-1)  # C
        max_index = origin_idx_calculator(y.max(-1)[1], B, H, W, num_chunks)
        min_index = origin_idx_calculator(y.min(-1)[1], B, H, W, num_chunks)
        scale_fix = 1 / ((2 * math.log(y.size(-1))) ** 0.5)
        # scale_fix = 1 / ((2 * math.log(num_chunks)) ** 0.5)
        scale = 1 / ((avg_max - avg_min) * scale_fix + eps) 
        # scale_ = 1 / ((avg_max - avg_min) * scale_fix) 

        avg = avg.view(1, -1, 1, 1)
        # scale_ = scale_.view(1, -1, 1, 1)
        scale = scale.view(1, -1, 1, 1)

        cache = x
        return avg, scale
    
    @staticmethod
    def backward(dout, cache):
        
        x = cache
        B, C, H, W = x.shape
        dL_davg = (dout).sum(dim=(0, 2, 3), keepdim=True)
        avg_pc = dL_davg / (B * H * W)
        
        
        return avg_pc

def Dec2Hex(decimal):
    # Pack the decimal value as a 32-bit single-precision float
    # print(decimal)
    binary = struct.pack('!f', decimal)
    # print(binary)

    # Unpack the binary representation and extract the hexadecimal value
    ieee754_hex = struct.unpack('!I', binary)[0]
    # print(ieee754_hex)

    # Convert the hexadecimal value to a string representation
    ieee754_hex_str = hex(ieee754_hex)[2:].zfill(8)
    # print(ieee754_hex_str)

    if(int(ieee754_hex_str[4],16)>7):
      b = int(ieee754_hex_str[0:4],16) + 1
    else:
      b = int(ieee754_hex_str[0:4],16)
    c = hex(b)[2:]

    return c.zfill(4)


def Mean_Var_Dec2Bfloat(Mean, Var, Exponent_Bit, Mantissa_Bit): 
    Mean = Mean.flatten().tolist()
    Var = Var.flatten().tolist()
    
    Mean_List=[]
    Var_List=[]
    Mean_List.clear()
    Var_List.clear()
    for mean in Mean: 
        Truncated_Rounded_Hex = Dec2Hex(mean)
        Mean_List.append(Truncated_Rounded_Hex)
    for var in Var: 
        Truncated_Rounded_Hex = Dec2Hex(var)
        Var_List.append(Truncated_Rounded_Hex)
    return Mean_List, Var_List


def Mean_Var_Dec2Bfloat_Back(Mean, Var, gamma, Exponent_Bit, Mantissa_Bit): 
    Mean = Mean.flatten().tolist()
    Var = Var.flatten().tolist()
    gamma = gamma.flatten().tolist()
    
    Mean_List=[]
    Var_List=[]
    gamma_List=[]
    Mean_List.clear()
    Var_List.clear()
    gamma_List.clear()
    for mean in Mean: 
        Truncated_Rounded_Hex = Dec2Hex(mean)
        Mean_List.append(Truncated_Rounded_Hex)
    for var in Var: 
        Truncated_Rounded_Hex = Dec2Hex(var)
        Var_List.append(Truncated_Rounded_Hex)
    for e in gamma: 
        Truncated_Rounded_Hex = Dec2Hex(e)
        gamma_List.append(Truncated_Rounded_Hex)    
    return Mean_List, Var_List, gamma_List


def Loss_Gradient_Dec2Bfloat(Loss_Gradient, Exponent_Bit, Mantissa_Bit): 
    Loss_Gradient = Loss_Gradient.flatten().tolist()
    Loss_Gradient_List=[]
    Loss_Gradient_List.clear()
    Zero_List = ["0000"]*(13*13*3)
    for loss_gradient in Loss_Gradient: 
        Truncated_Rounded_Hex = Dec2Hex(loss_gradient)
        Loss_Gradient_List.append(Truncated_Rounded_Hex)
    Loss_Gradient_List.extend(Zero_List)
    return Loss_Gradient_List

def Read_WeightGradient_Bfloat2Dec(Data_List_CH0, Data_List_CH1, Exponent_Bit, Mantissa_Bit, Out_CH, In_CH, Layer8=False):
    # Layer 8: Out_CH=128, Out_Size=13
    # Read all the Weights from a Weight_Folder
    Input_List0 = Data_List_CH0
    Input_List1 = Data_List_CH1
    
    Input_List = Weight_Gradient_Hardware_ReOrdering(Out_CH, In_CH, Input_List0, Input_List1)

    Out_List = []
    Out_List.clear()
    if Layer8:
        Out_List = Input_List[:(len(Input_List)-3*1024)]
    else: 
        Out_List = Input_List   
    return Out_List


def Read_WeightGradient_Bfloat2Dec_whole_image(Data_List_CH0, Data_List_CH1, Exponent_Bit, Mantissa_Bit, Out_CH, In_CH, Layer8=False):
    # Layer 8: Out_CH=128, Out_Size=13
    # Read all the Weights from a Weight_Folder
    Input_List0 = Data_List_CH0
    Input_List1 = Data_List_CH1
    
    
    #Input_List = Weight_Gradient_Hardware_ReOrdering_whole_image(Out_CH, In_CH, Input_List0, Input_List1)

    Out_List = []
    Out_List.clear()
    
    if Layer8:
        Input0, Input1, Input2, Input3, Input4, Input5, Input6, Input7  = Weight_Gradient_Hardware_ReOrdering_whole_image_layer8(Out_CH, In_CH, Input_List0, Input_List1)
        Out_List0 = Input0[:(len(Input0)-3*1024)]
        Out_List1 = Input1[:(len(Input1)-3*1024)]
        Out_List2 = Input2[:(len(Input2)-3*1024)]
        Out_List3 = Input3[:(len(Input3)-3*1024)]
        Out_List4 = Input4[:(len(Input4)-3*1024)]
        Out_List5 = Input5[:(len(Input5)-3*1024)]
        Out_List6 = Input6[:(len(Input6)-3*1024)]
        Out_List7 = Input7[:(len(Input7)-3*1024)]
        Out_List = [Out_List0, Out_List1, Out_List2, Out_List3, Out_List4, Out_List5, Out_List6, Out_List7]
        return Out_List
    else:
        Input_List = Weight_Gradient_Hardware_ReOrdering_whole_image(Out_CH, In_CH, Input_List0, Input_List1) 
        Out_List = [Input_List]  
        return Out_List


def Reversed_FlipHex_32To256(Hex32_List, segment_length=8):
    Reversed_Flip_List = Flip_Data(Hex32_List)
    combined_hex_list = []
    combined_hex_list.clear()
    for i in range(0, len(Reversed_Flip_List), segment_length):
        combined_hex = ''.join(Reversed_Flip_List[i:i+segment_length])
        combined_hex_list.append(combined_hex)
    return combined_hex_list

def Flip_Data(Data_List):
    Flip_Data_List = []
    for i in range(0, len(Data_List), 8):
        segment = Data_List[i:i + 8]
        if len(segment) == 8:
            reversed_segment = list(reversed(segment))
            Flip_Data_List.extend(reversed_segment)
    return Flip_Data_List

def Break_FlipHex_256To32(hex_strings, segment_length=8):
    segmented_str = []
    Hex32 = []
    segmented_str.clear()
    Hex32.clear()
    for hex_string in hex_strings:
        for value in hex_string:
            segments = [value[i:i+segment_length] for i in range(0, len(value), segment_length)]
            segmented_str.append(segments)
    for segments in segmented_str:
        for segment in segments:
            Hex32.append(segment)
    Flip_Data_List = Flip_Data(Hex32)
    return Flip_Data_List

def save_weights(weights, file_path):
    weights = weights.flatten().tolist()
    with open(file_path, 'w') as file:
        for weight in weights:
            file.write(f"{weight}\n")


def Convert_Tensor(Weight_List):
    weight_values = []
    for Value in Weight_List[0][:(len(Weight_List[0]))]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
        weight_values.append(str(Decimal))   
    Weight_Tensor0 = torch.tensor([float(value) for value in weight_values], dtype=torch.float32).view((16, 3, 3, 3))
    weight_values.clear()

    for Value in Weight_List[1][:(len(Weight_List[1]))]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
        weight_values.append(str(Decimal)) 
    Weight_Tensor1 = torch.tensor([float(value) for value in weight_values], dtype=torch.float32).view((32, 16, 3, 3))
    weight_values.clear()

    for Value in Weight_List[2][:(len(Weight_List[2]))]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
        weight_values.append(str(Decimal)) 
    Weight_Tensor2 = torch.tensor([float(value) for value in weight_values], dtype=torch.float32).view((64, 32, 3, 3))
    weight_values.clear()

    for Value in Weight_List[3][:(len(Weight_List[3]))]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
        weight_values.append(str(Decimal)) 
    Weight_Tensor3 = torch.tensor([float(value) for value in weight_values], dtype=torch.float32).view((128, 64, 3, 3))
    weight_values.clear()

    for Value in Weight_List[4][:(len(Weight_List[4]))]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
        weight_values.append(str(Decimal)) 
    Weight_Tensor4 = torch.tensor([float(value) for value in weight_values], dtype=torch.float32).view((256, 128, 3, 3))
    weight_values.clear()

    for Value in Weight_List[5][:(len(Weight_List[5]))]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
        weight_values.append(str(Decimal)) 
    Weight_Tensor5 = torch.tensor([float(value) for value in weight_values], dtype=torch.float32).view((512, 256, 3, 3))
    weight_values.clear()

    for Value in Weight_List[6][:(len(Weight_List[6]))]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
        weight_values.append(str(Decimal)) 
    Weight_Tensor6 = torch.tensor([float(value) for value in weight_values], dtype=torch.float32).view((1024, 512, 3, 3))
    weight_values.clear()

    for Value in Weight_List[7][:(len(Weight_List[7]))]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
        weight_values.append(str(Decimal)) 
    Weight_Tensor7 = torch.tensor([float(value) for value in weight_values], dtype=torch.float32).view((1024, 1024, 3, 3))
    weight_values.clear()

    for Value in Weight_List[8][:(len(Weight_List[8]))]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
        weight_values.append(str(Decimal)) 
    Weight_Tensor8 = torch.tensor([float(value) for value in weight_values], dtype=torch.float32).view((125, 1024, 1, 1))
    weight_values.clear()

    Weight_Tensor = [Weight_Tensor0, Weight_Tensor1, Weight_Tensor2, Weight_Tensor3, Weight_Tensor4, 
                     Weight_Tensor5, Weight_Tensor6, Weight_Tensor7, Weight_Tensor8]
    return Weight_Tensor


def Convert_BN_Tensor(BN_List):
    weight_values = []
    for Value in BN_List[0][:(len(BN_List[0]))]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
        weight_values.append(str(Decimal)) 
    BN_Tensor0 = torch.tensor([float(value) for value in weight_values], dtype=torch.float32).view((16,))
    weight_values.clear()

    for Value in BN_List[1][:(len(BN_List[1]))]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
        weight_values.append(str(Decimal)) 
    BN_Tensor1 = torch.tensor([float(value) for value in weight_values], dtype=torch.float32).view((32,))
    weight_values.clear()

    for Value in BN_List[2][:(len(BN_List[2]))]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
        weight_values.append(str(Decimal)) 
    BN_Tensor2 = torch.tensor([float(value) for value in weight_values], dtype=torch.float32).view((64,))
    weight_values.clear()

    for Value in BN_List[3][:(len(BN_List[3]))]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
        weight_values.append(str(Decimal)) 
    BN_Tensor3 = torch.tensor([float(value) for value in weight_values], dtype=torch.float32).view((128,))
    weight_values.clear()

    for Value in BN_List[4][:(len(BN_List[4]))]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
        weight_values.append(str(Decimal)) 
    BN_Tensor4 = torch.tensor([float(value) for value in weight_values], dtype=torch.float32).view((256,))
    weight_values.clear()

    for Value in BN_List[5][:(len(BN_List[5]))]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
        weight_values.append(str(Decimal)) 
    BN_Tensor5 = torch.tensor([float(value) for value in weight_values], dtype=torch.float32).view((512,))
    weight_values.clear()

    for Value in BN_List[6][:(len(BN_List[6]))]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
        weight_values.append(str(Decimal)) 
    BN_Tensor6 = torch.tensor([float(value) for value in weight_values], dtype=torch.float32).view((1024,))
    weight_values.clear()

    for Value in BN_List[7][:(len(BN_List[7]))]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bits, Mantissa_Bits)
        weight_values.append(str(Decimal)) 
    BN_Tensor7 = torch.tensor([float(value) for value in weight_values], dtype=torch.float32).view((1024,))
    weight_values.clear()
    
    BN_Tensor = [BN_Tensor0, BN_Tensor1, BN_Tensor2, BN_Tensor3, BN_Tensor4, BN_Tensor5, BN_Tensor6, BN_Tensor7]
    return BN_Tensor


'''
def Convert_Tensor(Weight_List):
    Weight_Tensor0 = torch.tensor(Weight_List[0], dtype=torch.float32).view((16, 3, 3, 3))
    Weight_Tensor1 = torch.tensor(Weight_List[1], dtype=torch.float32).view((32, 16, 3, 3))
    Weight_Tensor2 = torch.tensor(Weight_List[2], dtype=torch.float32).view((64, 32, 3, 3))
    Weight_Tensor3 = torch.tensor(Weight_List[3], dtype=torch.float32).view((128, 64, 3, 3))
    Weight_Tensor4 = torch.tensor(Weight_List[4], dtype=torch.float32).view((256, 128, 3, 3))
    Weight_Tensor5 = torch.tensor(Weight_List[5], dtype=torch.float32).view((512, 256, 3, 3))
    Weight_Tensor6 = torch.tensor(Weight_List[6], dtype=torch.float32).view((1024, 512, 3, 3))
    Weight_Tensor7 = torch.tensor(Weight_List[7], dtype=torch.float32).view((1024, 1024, 3, 3))
    Weight_Tensor8 = torch.tensor(Weight_List[8], dtype=torch.float32).view((125, 1024, 1, 1))
    Weight_Tensor = [Weight_Tensor0, Weight_Tensor1, Weight_Tensor2, Weight_Tensor3, Weight_Tensor4, 
                     Weight_Tensor5, Weight_Tensor6, Weight_Tensor7, Weight_Tensor8]
    return Weight_Tensor

def Convert_BN_Tensor(BN_List):
    BN_Tensor0 = torch.tensor(BN_List[0], dtype=torch.float32).view((16,))
    BN_Tensor1 = torch.tensor(BN_List[1], dtype=torch.float32).view((32,))
    BN_Tensor2 = torch.tensor(BN_List[2], dtype=torch.float32).view((64,))
    BN_Tensor3 = torch.tensor(BN_List[3], dtype=torch.float32).view((128,))
    BN_Tensor4 = torch.tensor(BN_List[4], dtype=torch.float32).view((256,))
    BN_Tensor5 = torch.tensor(BN_List[5], dtype=torch.float32).view((512,))
    BN_Tensor6 = torch.tensor(BN_List[6], dtype=torch.float32).view((1024,))
    BN_Tensor7 = torch.tensor(BN_List[7], dtype=torch.float32).view((1024,))
    BN_Tensor = [BN_Tensor0, BN_Tensor1, BN_Tensor2, BN_Tensor3, BN_Tensor4, BN_Tensor5, BN_Tensor6, BN_Tensor7]
    return BN_Tensor

def Convert_Bias_Tensor(Bias_List):
    Bias_Tensor = torch.tensor(Bias_List, dtype=torch.float32).view((125,))
    return Bias_Tensor
'''


# if __name__ == "__main__":
#     # Floating Point Parameters that we use
#     FloatingPoint_Format = "SinglePrecision"
#
#     # For the BFP16, we don't need to select format, we just use Single Precision, Truncated and Rounding
#
#     Selected_FP_Format = {
#         "SinglePrecision": (8, 23),
#         "HalfPrecision": (5, 10),
#         "Custom": (7, 8)
#     }
#
#     Exponent_Bits, Mantissa_Bits = Selected_FP_Format.get(FloatingPoint_Format, (0, 0))
#
#     preprocessor = PreProcessing(
#         Mode='Training',
#         VOC_Dataset=True,
#         Brain_Floating_Point=True,
#         Micro_code='micro_code',
#         Image_Converted=False,
#         Weight_Converted=True,
#         Bias_Converted=False,
#         BN_Param_Converted=False,
#         Exponent_Bits=8,
#         Mantissa_Bits=23
#     )
#
#     Weight_List = preprocessor.Weight_Converted_Func()
#     print(Weight_List[0])

'''
def BN(x, gamma, beta):
        
        gamma = gamma
        
        eps = 1e-5
        num_chunks = 8
        B, C, H, W = x.shape
        y = x.transpose(0, 1).contiguous()  # C x B x H x W
        y = y.view(C, num_chunks, B * H * W // num_chunks)
        avg_max = y.max(-1)[0].mean(-1)  # C
        avg_min = y.min(-1)[0].mean(-1)  # C
        avg = y.view(C, -1).mean(-1)  # C
        max_index = origin_idx_calculator(y.max(-1)[1], B, H, W, num_chunks)
        min_index = origin_idx_calculator(y.min(-1)[1], B, H, W, num_chunks)
        scale_fix = 1 / ((2 * math.log(y.size(-1))) ** 0.5)
        # scale_fix = 1 / ((2 * math.log(num_chunks)) ** 0.5)
        scale = 1 / ((avg_max - avg_min) * scale_fix + eps)  
        # scale = 1 / ((avg_max - avg_min) * scale_fix)  

        avg = avg.view(1, -1, 1, 1)
        scale = scale.view(1, -1, 1, 1)

        momentum = 0.1

        output = (x - avg) * scale

        output = output * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)

        cache = (x, gamma, beta, output, output_hat, scale, scale_fix, avg, avg_max, avg_min, eps, num_chunks, max_index, min_index)
        return cache
'''

'''
def BN(x, gamma, beta):
        
        gamma = gamma
        
        eps = 1e-5
        num_chunks = 8
        B, C, H, W = x.shape
        y = x.transpose(0, 1).contiguous()  # C x B x H x W
        y = y.view(C, num_chunks, B * H * W // num_chunks)
        avg_max = y.max(-1)[0].mean(-1)  # C
        avg_min = y.min(-1)[0].mean(-1)  # C
        # avg_max = y.max(-1)[0].max(-1)[0]  # C
        # avg_min = y.min(-1)[0].min(-1)[0]  # C
        avg = y.view(C, -1).mean(-1)  # C
        max_index = origin_idx_calculator(y.max(-1)[1], B, H, W, num_chunks)
        min_index = origin_idx_calculator(y.min(-1)[1], B, H, W, num_chunks)
        scale_fix = 1 / ((2 * math.log(y.size(-1))) ** 0.5)
        # scale_fix = 1 / ((2 * math.log(num_chunks)) ** 0.5)
        scale = 1 / ((avg_max - avg_min) * scale_fix + eps)  
        scale_ = 1 / ((avg_max - avg_min) * scale_fix)  

        avg = avg.view(1, -1, 1, 1)
        scale_ = scale_.view(1, -1, 1, 1)
        scale = scale.view(1, -1, 1, 1)

        momentum = 0.1

        output_hat = (x - avg) * scale_

        output = output_hat * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)

        cache = (x, gamma, beta, output, output_hat, scale, scale_fix, avg, avg_max, avg_min, eps, num_chunks, max_index, min_index)
        return cache
'''

'''
#latest
def BN(x, gamma, beta):
    out, cache = None, None
        
    eps = 1e-5
    D = gamma.shape[0]
    num_chunks = 8
    # running_mean = running_mean
    # running_var = running_var
    B, C, H, W = x.shape
    y = x.transpose(0, 1).contiguous()  # C x B x H x W
    y = y.view(C, num_chunks, B * H * W // num_chunks)
    avg_max = y.max(-1)[0].mean(-1)  # C
    avg_min = y.min(-1)[0].mean(-1)  # C
    avg = y.view(C, -1).mean(-1)  # C
    max_index = origin_idx_calculator(y.max(-1)[1], B, H, W, num_chunks)
    min_index = origin_idx_calculator(y.min(-1)[1], B, H, W, num_chunks)
    scale_fix = 1 / ((2 * math.log(y.size(-1))) ** 0.5)
    # scale = 1 / ((avg_max - avg_min) * scale_fix + eps)  
    scale_ = 1 / ((avg_max - avg_min) * scale_fix) 


    avg = avg.view(1, -1, 1, 1)
    scale_ = scale_.view(1, -1, 1, 1)
    
    momentum = 0.1

    output_hat = (x - avg) * scale_
    # ctx.save_for_backward(X, gamma, beta, output, scale)

    output = output_hat * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    
    cache = (x, gamma, beta, output_hat, scale_, scale_fix, avg, avg_max, avg_min, eps, num_chunks, max_index, min_index)
    
    return cache
'''

# def BN(x, gamma, beta):

#     out, cache = None, None
            
#     eps = 1e-5
#     D = gamma.shape[0]
#     num_chunks = 8
#     # running_mean = bn_params.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
#     # running_var = bn_params.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))
#     B, C, H, W = x.shape
#     # y = x.transpose(0, 1).contiguous()  # C x B x H x W
#     y = x.permute(1, 0, 2, 3).contiguous()  # C x B x H x W
#     y = y.view(C, num_chunks, B * H * W // num_chunks)
#     avg_max = y.max(-1)[0].mean(-1)  # C
#     avg_min = y.min(-1)[0].mean(-1)  # C
#     avg = y.view(C, -1).mean(-1)  # C
#     ## max_index = origin_idx_calculator(y.max(-1)[1], B, H, W, num_chunks)
#     ## min_index = origin_idx_calculator(y.min(-1)[1], B, H, W, num_chunks)
#     scale_fix = 1 / ((2 * math.log(y.size(-1))) ** 0.5)
#     scale = 1 / ((avg_max - avg_min) * scale_fix + eps)  

#     avg = avg.view(1, -1, 1, 1)
#     scale = scale.view(1, -1, 1, 1)
    
#     momentum = 0.1

#     output = (x - avg) * scale

#     output = output * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    
#     # running_mean = running_mean * momentum + (1 - momentum) * avg
#     # running_var = running_var * momentum + (1 - momentum) * scale
    
#     #cache = (x, gamma, beta, output, scale, scale_fix, avg, avg_max, avg_min, eps, num_chunks, max_index, min_index)
#     cache = (x, gamma, output, scale)
    
#     return cache


def BN(x, gamma, beta):

    out, cache = None, None
            
    eps = 1e-5
    D = gamma.shape[0]
    num_chunks = 8
    # running_mean = bn_params.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
    # running_var = bn_params.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))
    B, C, H, W = x.shape
    # y = x.transpose(0, 1).contiguous()  # C x B x H x W
    y = x.transpose(0, 1).contiguous()  # C x B x H x W
    y = y.view(C, num_chunks, B * H * W // num_chunks)
    avg_max = y.max(-1)[0].mean(-1)  # C
    avg_min = y.min(-1)[0].mean(-1)  # C
    avg = y.view(C, -1).mean(-1)  # C
    ## max_index = origin_idx_calculator(y.max(-1)[1], B, H, W, num_chunks)
    ## min_index = origin_idx_calculator(y.min(-1)[1], B, H, W, num_chunks)
    scale_fix = 1 / ((2 * math.log(y.size(-1))) ** 0.5)
    scale = 1 / ((avg_max - avg_min) * scale_fix + eps)  

    avg = avg.view(1, -1, 1, 1)
    scale = scale.view(1, -1, 1, 1)
    
    momentum = 0.1

    output = (x - avg) * scale

    output = output * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    
    # running_mean = running_mean * momentum + (1 - momentum) * avg
    # running_var = running_var * momentum + (1 - momentum) * scale
    
    #cache = (x, gamma, beta, output, scale, scale_fix, avg, avg_max, avg_min, eps, num_chunks, max_index, min_index)
    cache = (x, gamma, output, scale)
    
    return cache


def bfloat16_to_decimal(hex_str):
    # 32 비트 부동 소수점 값을 hex 문자열로 변환
    float32_hex = hex_str.ljust(8,'0')
    hex_data = bytes.fromhex(float32_hex)
    # hex 문자열을 부동 소수점 값으로 언팩
    decimal_value = struct.unpack('!f', hex_data)[0]

    return decimal_value

def Read_WeightGradient_Bfloat2Dec_Layer0(Data_List_CH0, Data_List_CH1, Exponent_Bit, Mantissa_Bit, Out_CH, In_CH, Layer8=False):
    # Layer 8: Out_CH=128, Out_Size=13
    # Read all the Weights from a Weight_Folder
    Input_List0 = Data_List_CH0
    Input_List1 = Data_List_CH1
    
    Input_List = Weight_Gradient_Hardware_ReOrdering_Layer0(Out_CH, In_CH, Input_List0, Input_List1)
    
    Out_List = []
    Out_List.clear()
    for Value in Input_List[:(len(Input_List))]:
        # Hexadecimal = str(Value) + "0000"
        # Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        # Decimal = Binary2Floating(Binary_Value, Exponent_Bit, Mantissa_Bit)
        Decimal = bfloat16_to_decimal(Value)
        Out_List.append(str(Decimal))   
    return Out_List


def Convert_Bias_Tensor(Bias_List):
    Bias_Tensor = torch.tensor(Bias_List, dtype=torch.float32).view((125,))
    return Bias_Tensor


# def Read_Weight_into_Bfloat16(Data_Tensor, Exponent_Bit, Mantissa_Bit): 
#     List_Sorted = Data_Tensor.flatten().tolist()
#     Input_List = [np.float32(Value) for Value in List_Sorted]
#     Hex_List = []
#     for Value in Input_List:
#         Binary_Value = Floating2Binary(Value, Exponent_Bit, Mantissa_Bit)
#         Hexadecimal_Value = hex(int(Binary_Value, 2))[2:]
#         Truncated_Rounded_Hex = Truncating_Rounding(Hexadecimal_Value)
#         Hex_List.append(Truncated_Rounded_Hex)
#     return Hex_List


def Read_Weight_into_Bfloat16(Data_Tensor, Exponent_Bit, Mantissa_Bit): 
    List_Sorted = Data_Tensor.flatten().tolist()
    origin_ar = np.array(List_Sorted)
    def decimal_to_ieee754(decimal):
        binary = struct.pack('!f', decimal)

        ieee754_hex = struct.unpack('!I', binary)[0]

        ieee754_hex_str = hex(ieee754_hex)[2:].zfill(8)

        if(int(ieee754_hex_str[4],16)>7):
            b = int(ieee754_hex_str[0:4],16) + 1
        else:
            b = int(ieee754_hex_str[0:4],16)
        c = hex(b)[2:]

        return c.zfill(4)
    trans_ar = np.vectorize(decimal_to_ieee754)(origin_ar)
    Hex_List = trans_ar.tolist()
    return Hex_List



def Read_Bias_into_Bfloat16(Data_List, Exponent_Bit, Mantissa_Bit):
    # Read all the Bias from a Bias_Folder

    List_Sorted = Data_List  # Sort the file names alphabetically

    Input_List = [np.float32(Value) for Value in List_Sorted]
    Hex_List = []
    for Value in Input_List:
        Binary_Value = Floating2Binary(Value, Exponent_Bit, Mantissa_Bit)
        Hexadecimal_Value = hex(int(Binary_Value, 2))[2:]
        Truncated_Rounded_Hex = Truncating_Rounding(Hexadecimal_Value)
        Hex_List.append(Truncated_Rounded_Hex)

    return Hex_List