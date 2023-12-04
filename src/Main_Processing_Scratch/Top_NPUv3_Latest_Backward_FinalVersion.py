import sys
sys.path.append("../")
from Pre_Processing_Scratch.Pre_Processing import Pre_Processing, Read_OutFmap_Bfloat2Dec, Cal_mean_var, Mean_Var_Dec2Bfloat, \
                                                Loss_Gradient_Dec2Bfloat, Read_WeightGradient_Bfloat2Dec, Break_FlipHex_256To32
from Pre_Processing_Scratch.Pre_Processing_Function import *
from Post_Processing_Scratch.Post_Processing_2Iterations import Post_Processing
import time
from tabulate import tabulate
from PCIe.PCIe import *

# 2023/08/21: Implemented By Thaising
# Combined Master-PhD in MSISLAB

# Floating Point Parameters that we use
FloatingPoint_Format = "FP32"

# For the BFP16, we don't need to select format, we just use Single Precision, Truncated and Rounding

Selected_FP_Format = {
    "FP32": (8, 23),
    "Bfloat16": (5, 10),
    "Custom": (7, 8)
}

Exponent_Bits, Mantissa_Bits = Selected_FP_Format.get(FloatingPoint_Format, (0, 0))

# Pre-define Conditions:
Mode = "Training"  # Declare whether is Training or Inference

# Pre-Processing Pre-Defined Conditions
Brain_Floating_Point    = True  # Declare Whether Bfloat16 Conversion or not
# Signals for Debugging Purposes
Image_Converted         = True  # Converted Images
Weight_Converted        = True  # Converted Weight
Bias_Converted          = True  # Converted Bias
BN_Param_Converted      = True  # Converted BN Parameters
Microcode               = True  # Microcode Writing

# Running Hardware Model:
YOLOv2_Hardware_Forward     = True
YOLOv2_Hardware_Backward    = True

######################################################################################################
#                                                                                                    #
#                                  Pre-Processing Forward                                            #
#                                                                                                    #
######################################################################################################

# Pre-Processing Class Initialization
PreProcessing = Pre_Processing(Mode                 =   Mode,
                               Brain_Floating_Point =   Brain_Floating_Point,
                               Exponent_Bits        =   Exponent_Bits,
                               Mantissa_Bits        =   Mantissa_Bits)

# Defined the Parameter for Each Variable for Pre-Processing
if Bias_Converted:
    Bias_List = PreProcessing.Bias_Converted_Func()
    
if BN_Param_Converted:
    Beta_List = PreProcessing.Beta_Param_Converted_Func()
    Gamma_List = PreProcessing.Gamma_Param_Converted_Func()
    RunningMean_List = PreProcessing.Running_Mean_Param_Converted_Func()
    RunningVar_List = PreProcessing.Running_Var_Param_Converted_Func()
    
if Weight_Converted:
    tic = time.time()
    print("\t --> " + " The Weights are processing, Please Wait for a Moment!")
    Weight_List = PreProcessing.Weight_Converted_Func()
    Weight_1st_Layer0 = New_Weight_Hardware_ReOrdering_Layer0(16, 16, Weight_List[0], ['0000']*16, ['0000']*16, ['0000']*16, Iteration="1")
    Weight_1st_Layer1 = New_Weight_Hardware_ReOrdering_OtherLayer(32, 16, Weight_List[1], ['0000']*32, ['0000']*32, ['0000']*32, Iteration="1")
    Weight_1st_Layer2 = New_Weight_Hardware_ReOrdering_OtherLayer(64, 32, Weight_List[2], ['0000']*64, ['0000']*64, ['0000']*64, Iteration="1")
    Weight_1st_Layer3 = New_Weight_Hardware_ReOrdering_OtherLayer(128, 64, Weight_List[3], ['0000']*128, ['0000']*128, ['0000']*128, Iteration="1")
    Weight_1st_Layer4 = New_Weight_Hardware_ReOrdering_OtherLayer(256, 128, Weight_List[4], ['0000']*256, ['0000']*256, ['0000']*256, Iteration="1")
    Weight_1st_Layer5 = New_Weight_Hardware_ReOrdering_OtherLayer(512, 256, Weight_List[5], ['0000']*512, ['0000']*512, ['0000']*512, Iteration="1")
    Weight_1st_Layer6 = New_Weight_Hardware_ReOrdering_OtherLayer(1024, 512, Weight_List[6], ['0000']*1024, ['0000']*1024, ['0000']*1024, Iteration="1")
    Weight_1st_Layer7 = New_Weight_Hardware_ReOrdering_OtherLayer(1024, 1024, Weight_List[7], ['0000']*1024, ['0000']*1024, ['0000']*1024, Iteration="1")
    Weight_1st_Layer8 = New_Weight_Hardware_ReOrdering_Layer8(128, 1024, Weight_List[8], Bias_List)
    
    # List for Each DDR Channels: 
    Weight_1st_CH0 = Weight_1st_Layer0[0] + Weight_1st_Layer1[0] + Weight_1st_Layer2[0] + Weight_1st_Layer3[0] + Weight_1st_Layer4[0] + \
                    Weight_1st_Layer5[0] + Weight_1st_Layer6[0] + Weight_1st_Layer7[0] + Weight_1st_Layer8[0]
    Weight_1st_CH1 = Weight_1st_Layer0[1] + Weight_1st_Layer1[1] + Weight_1st_Layer2[1] + Weight_1st_Layer3[1] + Weight_1st_Layer4[1] + \
                    Weight_1st_Layer5[1] + Weight_1st_Layer6[1] + Weight_1st_Layer7[1] + Weight_1st_Layer8[1]
    
    # Break 256To32 and Flip the Data: 
    Weight_1st_CH0 = Break_FlipHex_256To32(Weight_1st_CH0)
    Weight_1st_CH1 = Break_FlipHex_256To32(Weight_1st_CH1)
                    
    # Write Weights into DDR: 
    PCIe2DDR(Weight_1st_CH0, Wr_Address=0x0)
    print("\t --> " + " Write Weight_1st_CH0 to DDR Done!")
    PCIe2DDR(Weight_1st_CH1, Wr_Address=0x8000000)
    print("\t --> " + " Write Weight_1st_CH1 to DDR Done!")
    print("\t --> " + " The Forward Weights are Successfully Converted and Written to DDR!")
    toc = time.time()
    cost_time1 = (toc-tic)
    
if Image_Converted:
    tic = time.time()
    print("\t --> " + " The Image is selecting and Converting!")
    Image_Path_CH0 = "/media/msis/MSIS/Python/NPUv3_Hardware_2Iterations/Pre_Processing_Scratch/Images/Image_CH0.mem"
    Image_Path_CH1 = "/media/msis/MSIS/Python/NPUv3_Hardware_2Iterations/Pre_Processing_Scratch/Images/Image_CH1.mem"

    Image_CH0 = []
    with open(Image_Path_CH0, mode="r") as data:
        data = data.readlines()
    Image_CH0 = [value.strip() for value in data]

    Image_CH1 = []
    with open(Image_Path_CH1, mode="r") as data:
        data = data.readlines()
    Image_CH1 = [value.strip() for value in data]
    
    # Write Weights into DDR:     
    PCIe2DDR(Image_CH0, Wr_Address=None)
    print("\t --> " + " Write Images_CH0 to DDR Done!")
    PCIe2DDR(Image_CH1, Wr_Address=None)
    print("\t --> " + " Write Images_CH1 to DDR Done!")
    
    print("\t --> " + " The Images are Successfully Converted and Written to DDR!")
    toc = time.time()
    cost_time2 = (toc-tic)
    print("Image Time: "+str(cost_time2/60) + " mn")

print("=== Our Pre-Processing is finished! ===\n\n=== Let's Starting Hardware Model Processing: ===")

#####################################################################################################
#                                                                                                   #
#                                         Microcode Writing                                         #
#                                                                                                   #
##################################################################################################### 
# Microcode including Forward + Backward
if Microcode:
    None 

if YOLOv2_Hardware_Forward:
    Pre_Processing_Time = cost_time1+cost_time2

    print("\n")
    print(tabulate([['8Images', cost_time2], ['Weights', cost_time1],
                    ['---------------------------------', str('----------------------')],
                    ['Total Spending Time', Pre_Processing_Time]], 
                    headers=['Python Pre-Processing', 'Total Time (s)'], tablefmt='orgtbl'))
    print("\n")

# Hardware Initialization
if YOLOv2_Hardware_Forward:
    tic = time.time()
    #####################################################################################################
    #                                                                                                   #
    #                                         Layer0_Forward                                            #
    #                                                                                                   #
    #####################################################################################################  
    ###########################################1st Mean/Var##############################################
    # Reading Output_1st_Layer0:
    OutImage1_1st_Layer0_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage1_1st_Layer0_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_1st_Layer0_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_1st_Layer0_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_1st_Layer0_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_1st_Layer0_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_1st_Layer0_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_1st_Layer0_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_1st_Layer0_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_1st_Layer0_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_1st_Layer0_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_1st_Layer0_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_1st_Layer0_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_1st_Layer0_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_1st_Layer0_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_1st_Layer0_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage1_1st_Layer0 = Read_OutFmap_Bfloat2Dec(OutImage1_1st_Layer0_CH0, OutImage1_1st_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    OutImage2_1st_Layer0 = Read_OutFmap_Bfloat2Dec(OutImage2_1st_Layer0_CH0, OutImage2_1st_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    OutImage3_1st_Layer0 = Read_OutFmap_Bfloat2Dec(OutImage3_1st_Layer0_CH0, OutImage3_1st_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    OutImage4_1st_Layer0 = Read_OutFmap_Bfloat2Dec(OutImage4_1st_Layer0_CH0, OutImage4_1st_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    OutImage5_1st_Layer0 = Read_OutFmap_Bfloat2Dec(OutImage5_1st_Layer0_CH0, OutImage5_1st_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    OutImage6_1st_Layer0 = Read_OutFmap_Bfloat2Dec(OutImage6_1st_Layer0_CH0, OutImage6_1st_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    OutImage7_1st_Layer0 = Read_OutFmap_Bfloat2Dec(OutImage7_1st_Layer0_CH0, OutImage7_1st_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    OutImage8_1st_Layer0 = Read_OutFmap_Bfloat2Dec(OutImage8_1st_Layer0_CH0, OutImage8_1st_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
    OutImages_1st_Layer0 = OutImage1_1st_Layer0 + OutImage2_1st_Layer0 + OutImage3_1st_Layer0 + OutImage4_1st_Layer0 + \
                            OutImage5_1st_Layer0 + OutImage6_1st_Layer0 + OutImage7_1st_Layer0 + OutImage8_1st_Layer0    
    OutImage_1st_Layer0 = torch.tensor([float(value) for value in OutImages_1st_Layer0], dtype=torch.float32).reshape(8, 16, 208, 208)
    
    # Calculate Mean/Var 
    Mean_1st_Layer0, Var_1st_Layer0 = Cal_mean_var.forward(OutImage_1st_Layer0)
    Mean_1st_Layer0, Var_1st_Layer0 = Mean_Var_Dec2Bfloat(Mean_1st_Layer0, Var_1st_Layer0, Exponent_Bits, Mantissa_Bits)
    Weight_2nd_Layer0 = New_Weight_Hardware_ReOrdering_Layer0(16, 16, Weight_List[0], Mean_1st_Layer0, Var_1st_Layer0, Beta_List[0], Iteration="2")
    print("Weight_2nd_Layer0_CH0: " + str(len(Weight_2nd_Layer0[0])))
    print("Weight_2nd_Layer0_CH1: " + str(len(Weight_2nd_Layer0[1])))
    
    # Write Weight_2nd_Layer0 to DDR: 
    PCIe2DDR(Weight_2nd_Layer0[0], Wr_Address=None)
    print("\t --> " + " Write Weight_2nd_Layer0_CH0 Done!")
    PCIe2DDR(Weight_2nd_Layer0[1], Wr_Address=None)
    print("\t --> " + " Write Weight_2nd_Layer0_CH1 Done!")    
    ###########################################2nd Iteration#############################################

    #####################################################################################################
    #                                                                                                   #
    #                                         Layer1_Forward                                            #
    #                                                                                                   #
    #####################################################################################################  
    ###########################################1st Mean/Var#####################################
    # Reading Output_1st_Layer1:
    OutImage1_1st_Layer1_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage1_1st_Layer1_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_1st_Layer1_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_1st_Layer1_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_1st_Layer1_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_1st_Layer1_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_1st_Layer1_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_1st_Layer1_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_1st_Layer1_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_1st_Layer1_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_1st_Layer1_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_1st_Layer1_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_1st_Layer1_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_1st_Layer1_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_1st_Layer1_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_1st_Layer1_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage1_1st_Layer1 = Read_OutFmap_Bfloat2Dec(OutImage1_1st_Layer1_CH0, OutImage1_1st_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    OutImage2_1st_Layer1 = Read_OutFmap_Bfloat2Dec(OutImage2_1st_Layer1_CH0, OutImage2_1st_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    OutImage3_1st_Layer1 = Read_OutFmap_Bfloat2Dec(OutImage3_1st_Layer1_CH0, OutImage3_1st_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    OutImage4_1st_Layer1 = Read_OutFmap_Bfloat2Dec(OutImage4_1st_Layer1_CH0, OutImage4_1st_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    OutImage5_1st_Layer1 = Read_OutFmap_Bfloat2Dec(OutImage5_1st_Layer1_CH0, OutImage5_1st_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    OutImage6_1st_Layer1 = Read_OutFmap_Bfloat2Dec(OutImage6_1st_Layer1_CH0, OutImage6_1st_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    OutImage7_1st_Layer1 = Read_OutFmap_Bfloat2Dec(OutImage7_1st_Layer1_CH0, OutImage7_1st_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    OutImage8_1st_Layer1 = Read_OutFmap_Bfloat2Dec(OutImage8_1st_Layer1_CH0, OutImage8_1st_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
    OutImages_1st_Layer1 = OutImage1_1st_Layer1 + OutImage2_1st_Layer1 + OutImage3_1st_Layer1 + OutImage4_1st_Layer1 + \
                            OutImage5_1st_Layer1 + OutImage6_1st_Layer1 + OutImage7_1st_Layer1 + OutImage8_1st_Layer1    
    OutImage_1st_Layer1 = torch.tensor([float(value) for value in OutImages_1st_Layer1], dtype=torch.float32).reshape(8, 32, 104, 104)
    
    # Calculate Mean/Var: 
    Mean_1st_Layer1, Var_1st_Layer1 = Cal_mean_var.forward(OutImage_1st_Layer1)
    Mean_1st_Layer1, Var_1st_Layer1 = Mean_Var_Dec2Bfloat(Mean_1st_Layer1, Var_1st_Layer1, Exponent_Bits, Mantissa_Bits)
    Weight_2nd_Layer1 = New_Weight_Hardware_ReOrdering_OtherLayer(32, 16, Weight_List[1], Mean_1st_Layer1, Var_1st_Layer1, Beta_List[1], Iteration="2")
    print("Weight_2nd_Layer1_CH0: " + str(len(Weight_2nd_Layer1[0])))
    print("Weight_2nd_Layer1_CH1: " + str(len(Weight_2nd_Layer1[1])))
    
    # Write Weight_2nd_Layer1 to DDR: 
    Weight_2nd_Layer1_CH0 = Break_FlipHex_256To32(Weight_2nd_Layer1[0])
    Weight_2nd_Layer1_CH1 = Break_FlipHex_256To32(Weight_2nd_Layer1[1])
    
    # Write Weight_2nd_Layer1 to DDR: 
    PCIe2DDR(Weight_2nd_Layer1_CH0, Wr_Address=None)
    print("\t --> " + " Write Weight_2nd_Layer1_CH0 Done!")
    PCIe2DDR(Weight_2nd_Layer1_CH1, Wr_Address=None)
    print("\t --> " + " Write Weight_2nd_Layer1_CH1 Done!")    
    ###########################################2nd Iteration#############################################

    #####################################################################################################
    #                                                                                                   #
    #                                         Layer2_Forward                                            #
    #                                                                                                   #
    #####################################################################################################  
    ###########################################1st Mean/Var#####################################
    # Reading Output_1st_Layer2:
    OutImage1_1st_Layer2_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage1_1st_Layer2_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_1st_Layer2_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_1st_Layer2_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_1st_Layer2_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_1st_Layer2_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_1st_Layer2_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_1st_Layer2_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_1st_Layer2_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_1st_Layer2_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_1st_Layer2_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_1st_Layer2_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_1st_Layer2_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_1st_Layer2_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_1st_Layer2_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_1st_Layer2_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage1_1st_Layer2 = Read_OutFmap_Bfloat2Dec(OutImage1_1st_Layer2_CH0, OutImage1_1st_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    OutImage2_1st_Layer2 = Read_OutFmap_Bfloat2Dec(OutImage2_1st_Layer2_CH0, OutImage2_1st_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    OutImage3_1st_Layer2 = Read_OutFmap_Bfloat2Dec(OutImage3_1st_Layer2_CH0, OutImage3_1st_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    OutImage4_1st_Layer2 = Read_OutFmap_Bfloat2Dec(OutImage4_1st_Layer2_CH0, OutImage4_1st_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    OutImage5_1st_Layer2 = Read_OutFmap_Bfloat2Dec(OutImage5_1st_Layer2_CH0, OutImage5_1st_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    OutImage6_1st_Layer2 = Read_OutFmap_Bfloat2Dec(OutImage6_1st_Layer2_CH0, OutImage6_1st_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    OutImage7_1st_Layer2 = Read_OutFmap_Bfloat2Dec(OutImage7_1st_Layer2_CH0, OutImage7_1st_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    OutImage8_1st_Layer2 = Read_OutFmap_Bfloat2Dec(OutImage8_1st_Layer2_CH0, OutImage8_1st_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
    OutImages_1st_Layer2 = OutImage1_1st_Layer2 + OutImage2_1st_Layer2 + OutImage3_1st_Layer2 + OutImage4_1st_Layer2 + \
                            OutImage5_1st_Layer2 + OutImage6_1st_Layer2 + OutImage7_1st_Layer2 + OutImage8_1st_Layer2    
    OutImage_1st_Layer2 = torch.tensor([float(value) for value in OutImages_1st_Layer2], dtype=torch.float32).reshape(8, 64, 52, 52)
    # Calculate Mean/Var 
    Mean_1st_Layer2, Var_1st_Layer2 = Cal_mean_var.forward(OutImage_1st_Layer2)
    Mean_1st_Layer2, Var_1st_Layer2 = Mean_Var_Dec2Bfloat(Mean_1st_Layer2, Var_1st_Layer2, Exponent_Bits, Mantissa_Bits)
    Weight_2nd_Layer2 = New_Weight_Hardware_ReOrdering_OtherLayer(64, 32, Weight_List[2], Mean_1st_Layer2, Var_1st_Layer2, Beta_List[2], Iteration="2")
    print("Weight_2nd_Layer2_CH0: " + str(len(Weight_2nd_Layer2[0])))
    print("Weight_2nd_Layer2_CH1: " + str(len(Weight_2nd_Layer2[1])))
    
    # Write Weight_2nd_Layer2 to DDR: 
    Weight_2nd_Layer2_CH0 = Break_FlipHex_256To32(Weight_2nd_Layer2[0])
    Weight_2nd_Layer2_CH1 = Break_FlipHex_256To32(Weight_2nd_Layer2[1])
    
    # Write Weight_2nd_Layer2 to DDR: 
    PCIe2DDR(Weight_2nd_Layer2_CH0, Wr_Address=None)
    print("\t --> " + " Write Weight_2nd_Layer2_CH0 Done!")
    PCIe2DDR(Weight_2nd_Layer2_CH1, Wr_Address=None)
    print("\t --> " + " Write Weight_2nd_Layer2_CH1 Done!")    
    ###########################################2nd Iteration#############################################

    #####################################################################################################
    #                                                                                                   #
    #                                         Layer3_Forward                                            #
    #                                                                                                   #
    #####################################################################################################  
    ###########################################1st Mean/Var#####################################
    # Reading Output_1st_Layer3:
    OutImage1_1st_Layer3_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage1_1st_Layer3_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_1st_Layer3_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_1st_Layer3_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_1st_Layer3_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_1st_Layer3_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_1st_Layer3_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_1st_Layer3_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_1st_Layer3_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_1st_Layer3_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_1st_Layer3_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_1st_Layer3_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_1st_Layer3_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_1st_Layer3_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_1st_Layer3_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_1st_Layer3_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage1_1st_Layer3 = Read_OutFmap_Bfloat2Dec(OutImage1_1st_Layer3_CH0, OutImage1_1st_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    OutImage2_1st_Layer3 = Read_OutFmap_Bfloat2Dec(OutImage2_1st_Layer3_CH0, OutImage2_1st_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    OutImage3_1st_Layer3 = Read_OutFmap_Bfloat2Dec(OutImage3_1st_Layer3_CH0, OutImage3_1st_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    OutImage4_1st_Layer3 = Read_OutFmap_Bfloat2Dec(OutImage4_1st_Layer3_CH0, OutImage4_1st_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    OutImage5_1st_Layer3 = Read_OutFmap_Bfloat2Dec(OutImage5_1st_Layer3_CH0, OutImage5_1st_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    OutImage6_1st_Layer3 = Read_OutFmap_Bfloat2Dec(OutImage6_1st_Layer3_CH0, OutImage6_1st_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    OutImage7_1st_Layer3 = Read_OutFmap_Bfloat2Dec(OutImage7_1st_Layer3_CH0, OutImage7_1st_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    OutImage8_1st_Layer3 = Read_OutFmap_Bfloat2Dec(OutImage8_1st_Layer3_CH0, OutImage8_1st_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
    OutImages_1st_Layer3 = OutImage1_1st_Layer3 + OutImage2_1st_Layer3 + OutImage3_1st_Layer3 + OutImage4_1st_Layer3 + \
                            OutImage5_1st_Layer3 + OutImage6_1st_Layer3 + OutImage7_1st_Layer3 + OutImage8_1st_Layer3    
    OutImage_1st_Layer3 = torch.tensor([float(value) for value in OutImages_1st_Layer3], dtype=torch.float32).reshape(8, 128, 26, 26)
    # Calculate Mean/Var 
    Mean_1st_Layer3, Var_1st_Layer3 = Cal_mean_var.forward(OutImage_1st_Layer3)
    Mean_1st_Layer3, Var_1st_Layer3 = Mean_Var_Dec2Bfloat(Mean_1st_Layer3, Var_1st_Layer3, Exponent_Bits, Mantissa_Bits)
    Weight_2nd_Layer3 = New_Weight_Hardware_ReOrdering_OtherLayer(128, 64, Weight_List[3], Mean_1st_Layer3, Var_1st_Layer3, Beta_List[3], Iteration="2")
    print("Weight_2nd_Layer3_CH0: " + str(len(Weight_2nd_Layer3[0])))
    print("Weight_2nd_Layer3_CH1: " + str(len(Weight_2nd_Layer3[1])))
    
    # Write Weight_2nd_Layer3 to DDR: 
    Weight_2nd_Layer3_CH0 = Break_FlipHex_256To32(Weight_2nd_Layer3[0])
    Weight_2nd_Layer3_CH1 = Break_FlipHex_256To32(Weight_2nd_Layer3[1])
    
    # Write Weight_2nd_Layer3 to DDR: 
    PCIe2DDR(Weight_2nd_Layer3_CH0, Wr_Address=None)
    print("\t --> " + " Write Weight_2nd_Layer3_CH0 Done!")
    PCIe2DDR(Weight_2nd_Layer3_CH1, Wr_Address=None)
    print("\t --> " + " Write Weight_2nd_Layer3_CH1 Done!")    
    ###########################################2nd Iteration#############################################

    #####################################################################################################
    #                                                                                                   #
    #                                         Layer4_Forward                                            #
    #                                                                                                   #
    #####################################################################################################  
    ###########################################1st Mean/Var#####################################
    # Reading Output_1st_Layer4:
    OutImage1_1st_Layer4_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage1_1st_Layer4_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_1st_Layer4_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_1st_Layer4_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_1st_Layer4_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_1st_Layer4_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_1st_Layer4_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_1st_Layer4_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_1st_Layer4_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_1st_Layer4_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_1st_Layer4_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_1st_Layer4_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_1st_Layer4_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_1st_Layer4_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_1st_Layer4_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_1st_Layer4_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage1_1st_Layer4 = Read_OutFmap_Bfloat2Dec(OutImage1_1st_Layer4_CH0, OutImage1_1st_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    OutImage2_1st_Layer4 = Read_OutFmap_Bfloat2Dec(OutImage2_1st_Layer4_CH0, OutImage2_1st_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    OutImage3_1st_Layer4 = Read_OutFmap_Bfloat2Dec(OutImage3_1st_Layer4_CH0, OutImage3_1st_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    OutImage4_1st_Layer4 = Read_OutFmap_Bfloat2Dec(OutImage4_1st_Layer4_CH0, OutImage4_1st_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    OutImage5_1st_Layer4 = Read_OutFmap_Bfloat2Dec(OutImage5_1st_Layer4_CH0, OutImage5_1st_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    OutImage6_1st_Layer4 = Read_OutFmap_Bfloat2Dec(OutImage6_1st_Layer4_CH0, OutImage6_1st_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    OutImage7_1st_Layer4 = Read_OutFmap_Bfloat2Dec(OutImage7_1st_Layer4_CH0, OutImage7_1st_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    OutImage8_1st_Layer4 = Read_OutFmap_Bfloat2Dec(OutImage8_1st_Layer4_CH0, OutImage8_1st_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
    OutImages_1st_Layer4 = OutImage1_1st_Layer4 + OutImage2_1st_Layer4 + OutImage3_1st_Layer4 + OutImage4_1st_Layer4 + \
                            OutImage5_1st_Layer4 + OutImage6_1st_Layer4 + OutImage7_1st_Layer4 + OutImage8_1st_Layer4    
    OutImage_1st_Layer4 = torch.tensor([float(value) for value in OutImages_1st_Layer4], dtype=torch.float32).reshape(8, 256, 13, 13)
    
    # Calculate Mean/Var 
    Mean_1st_Layer4, Var_1st_Layer4 = Cal_mean_var.forward(OutImage_1st_Layer4)
    Mean_1st_Layer4, Var_1st_Layer4 = Mean_Var_Dec2Bfloat(Mean_1st_Layer4, Var_1st_Layer4, Exponent_Bits, Mantissa_Bits)
    Weight_2nd_Layer4 = New_Weight_Hardware_ReOrdering_OtherLayer(256, 128, Weight_List[4], Mean_1st_Layer4, Var_1st_Layer4, Beta_List[4], Iteration="2")
    print("Weight_2nd_Layer4_CH0: " + str(len(Weight_2nd_Layer4[0])))
    print("Weight_2nd_Layer4_CH1: " + str(len(Weight_2nd_Layer4[1])))
    
    # Write Weight_2nd_Layer4 to DDR: 
    Weight_2nd_Layer4_CH0 = Break_FlipHex_256To32(Weight_2nd_Layer4[0])
    Weight_2nd_Layer4_CH1 = Break_FlipHex_256To32(Weight_2nd_Layer4[1])
    
    # Write Weight_2nd_Layer4 to DDR: 
    PCIe2DDR(Weight_2nd_Layer4[0], Wr_Address=None)
    print("\t --> " + " Write Weight_2nd_Layer4_CH0 Done!")
    PCIe2DDR(Weight_2nd_Layer4[1], Wr_Address=None)
    print("\t --> " + " Write Weight_2nd_Layer4_CH1 Done!")    
    ###########################################2nd Iteration#############################################

    #####################################################################################################
    #                                                                                                   #
    #                                         Layer5_Forward                                            #
    #                                                                                                   #
    #####################################################################################################  
    ###########################################1st Mean/Var#####################################
    # Reading Output_1st_Layer5:
    OutImage1_1st_Layer5_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage1_1st_Layer5_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_1st_Layer5_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_1st_Layer5_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_1st_Layer5_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_1st_Layer5_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_1st_Layer5_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_1st_Layer5_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_1st_Layer5_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_1st_Layer5_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_1st_Layer5_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_1st_Layer5_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_1st_Layer5_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_1st_Layer5_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_1st_Layer5_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_1st_Layer5_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage1_1st_Layer5 = Read_OutFmap_Bfloat2Dec(OutImage1_1st_Layer5_CH0, OutImage1_1st_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    OutImage2_1st_Layer5 = Read_OutFmap_Bfloat2Dec(OutImage2_1st_Layer5_CH0, OutImage2_1st_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    OutImage3_1st_Layer5 = Read_OutFmap_Bfloat2Dec(OutImage3_1st_Layer5_CH0, OutImage3_1st_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    OutImage4_1st_Layer5 = Read_OutFmap_Bfloat2Dec(OutImage4_1st_Layer5_CH0, OutImage4_1st_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    OutImage5_1st_Layer5 = Read_OutFmap_Bfloat2Dec(OutImage5_1st_Layer5_CH0, OutImage5_1st_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    OutImage6_1st_Layer5 = Read_OutFmap_Bfloat2Dec(OutImage6_1st_Layer5_CH0, OutImage6_1st_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    OutImage7_1st_Layer5 = Read_OutFmap_Bfloat2Dec(OutImage7_1st_Layer5_CH0, OutImage7_1st_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    OutImage8_1st_Layer5 = Read_OutFmap_Bfloat2Dec(OutImage8_1st_Layer5_CH0, OutImage8_1st_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
    OutImages_1st_Layer5 = OutImage1_1st_Layer5 + OutImage2_1st_Layer5 + OutImage3_1st_Layer5 + OutImage4_1st_Layer5 + \
                            OutImage5_1st_Layer5 + OutImage6_1st_Layer5 + OutImage7_1st_Layer5 + OutImage8_1st_Layer5    
    OutImage_1st_Layer5 = torch.tensor([float(value) for value in OutImages_1st_Layer5], dtype=torch.float32).reshape(8, 512, 13, 13)
    
    # Calculate Mean/Var 
    Mean_1st_Layer5, Var_1st_Layer5 = Cal_mean_var.forward(OutImage_1st_Layer5)
    Mean_1st_Layer5, Var_1st_Layer5 = Mean_Var_Dec2Bfloat(Mean_1st_Layer5, Var_1st_Layer5, Exponent_Bits, Mantissa_Bits)
    Weight_2nd_Layer5 = New_Weight_Hardware_ReOrdering_OtherLayer(512, 256, Weight_List[5], Mean_1st_Layer5, Var_1st_Layer5, Beta_List[5], Iteration="2")
    print("Weight_2nd_Layer5_CH0: " + str(len(Weight_2nd_Layer5[0])))
    print("Weight_2nd_Layer5_CH1: " + str(len(Weight_2nd_Layer5[1])))
    
    # Write Weight_2nd_Layer5 to DDR: 
    Weight_2nd_Layer5_CH0 = Break_FlipHex_256To32(Weight_2nd_Layer5[0])
    Weight_2nd_Layer5_CH1 = Break_FlipHex_256To32(Weight_2nd_Layer5[1])
    
    # Write Weight_2nd_Layer5 to DDR: 
    PCIe2DDR(Weight_2nd_Layer5_CH0, Wr_Address=None)
    print("\t --> " + " Write Weight_2nd_Layer5_CH0 Done!")
    PCIe2DDR(Weight_2nd_Layer5_CH1, Wr_Address=None)
    print("\t --> " + " Write Weight_2nd_Layer5_CH1 Done!")    
    ###########################################2nd Iteration#############################################

    #####################################################################################################
    #                                                                                                   #
    #                                         Layer6_Forward                                            #
    #                                                                                                   #
    #####################################################################################################  
    ###########################################1st Mean/Var#####################################
    # Reading Output_1st_Layer6:
    OutImage1_1st_Layer6_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage1_1st_Layer6_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_1st_Layer6_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_1st_Layer6_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_1st_Layer6_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_1st_Layer6_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_1st_Layer6_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_1st_Layer6_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_1st_Layer6_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_1st_Layer6_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_1st_Layer6_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_1st_Layer6_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_1st_Layer6_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_1st_Layer6_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_1st_Layer6_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_1st_Layer6_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage1_1st_Layer6 = Read_OutFmap_Bfloat2Dec(OutImage1_1st_Layer6_CH0, OutImage1_1st_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    OutImage2_1st_Layer6 = Read_OutFmap_Bfloat2Dec(OutImage2_1st_Layer6_CH0, OutImage2_1st_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    OutImage3_1st_Layer6 = Read_OutFmap_Bfloat2Dec(OutImage3_1st_Layer6_CH0, OutImage3_1st_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    OutImage4_1st_Layer6 = Read_OutFmap_Bfloat2Dec(OutImage4_1st_Layer6_CH0, OutImage4_1st_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    OutImage5_1st_Layer6 = Read_OutFmap_Bfloat2Dec(OutImage5_1st_Layer6_CH0, OutImage5_1st_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    OutImage6_1st_Layer6 = Read_OutFmap_Bfloat2Dec(OutImage6_1st_Layer6_CH0, OutImage6_1st_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    OutImage7_1st_Layer6 = Read_OutFmap_Bfloat2Dec(OutImage7_1st_Layer6_CH0, OutImage7_1st_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    OutImage8_1st_Layer6 = Read_OutFmap_Bfloat2Dec(OutImage8_1st_Layer6_CH0, OutImage8_1st_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    OutImages_1st_Layer6 = OutImage1_1st_Layer6 + OutImage2_1st_Layer6 + OutImage3_1st_Layer6 + OutImage4_1st_Layer6 + \
                            OutImage5_1st_Layer6 + OutImage6_1st_Layer6 + OutImage7_1st_Layer6 + OutImage8_1st_Layer6    
    OutImage_1st_Layer6 = torch.tensor([float(value) for value in OutImages_1st_Layer6], dtype=torch.float32).reshape(8, 1024, 13, 13)
    
    # Calculate Mean/Var 
    Mean_1st_Layer6, Var_1st_Layer6 = Cal_mean_var.forward(OutImage_1st_Layer6)
    Mean_1st_Layer6, Var_1st_Layer6 = Mean_Var_Dec2Bfloat(Mean_1st_Layer6, Var_1st_Layer6, Exponent_Bits, Mantissa_Bits)
    Weight_2nd_Layer6 = New_Weight_Hardware_ReOrdering_OtherLayer(1024, 512, Weight_List[6], Mean_1st_Layer6, Var_1st_Layer6, Beta_List[6], Iteration="2")
    print("Weight_2nd_Layer6_CH0: " + str(len(Weight_2nd_Layer6[0])))
    print("Weight_2nd_Layer6_CH1: " + str(len(Weight_2nd_Layer6[1])))
    
    # Write Weight_2nd_Layer6 to DDR: 
    Weight_2nd_Layer6_CH0 = Break_FlipHex_256To32(Weight_2nd_Layer6[0])
    Weight_2nd_Layer6_CH1 = Break_FlipHex_256To32(Weight_2nd_Layer6[1])
    
    # Write Weight_2nd_Layer6 to DDR: 
    PCIe2DDR(Weight_2nd_Layer6_CH0, Wr_Address=None)
    print("\t --> " + " Write Weight_2nd_Layer6_CH0 Done!")
    PCIe2DDR(Weight_2nd_Layer6_CH1, Wr_Address=None)
    print("\t --> " + " Write Weight_2nd_Layer6_CH1 Done!")    
    ###########################################2nd Iteration#############################################

    #####################################################################################################
    #                                                                                                   #
    #                                         Layer7_Forward                                            #
    #                                                                                                   #
    #####################################################################################################  
    ###########################################1st Mean/Var#####################################
    # Reading Output_1st_Layer7:
    OutImage1_1st_Layer7_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage1_1st_Layer7_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_1st_Layer7_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_1st_Layer7_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_1st_Layer7_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_1st_Layer7_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_1st_Layer7_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_1st_Layer7_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_1st_Layer7_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_1st_Layer7_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_1st_Layer7_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_1st_Layer7_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_1st_Layer7_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_1st_Layer7_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_1st_Layer7_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_1st_Layer7_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage1_1st_Layer7 = Read_OutFmap_Bfloat2Dec(OutImage1_1st_Layer7_CH0, OutImage1_1st_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    OutImage2_1st_Layer7 = Read_OutFmap_Bfloat2Dec(OutImage2_1st_Layer7_CH0, OutImage2_1st_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    OutImage3_1st_Layer7 = Read_OutFmap_Bfloat2Dec(OutImage3_1st_Layer7_CH0, OutImage3_1st_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    OutImage4_1st_Layer7 = Read_OutFmap_Bfloat2Dec(OutImage4_1st_Layer7_CH0, OutImage4_1st_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    OutImage5_1st_Layer7 = Read_OutFmap_Bfloat2Dec(OutImage5_1st_Layer7_CH0, OutImage5_1st_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    OutImage6_1st_Layer7 = Read_OutFmap_Bfloat2Dec(OutImage6_1st_Layer7_CH0, OutImage6_1st_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    OutImage7_1st_Layer7 = Read_OutFmap_Bfloat2Dec(OutImage7_1st_Layer7_CH0, OutImage7_1st_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    OutImage8_1st_Layer7 = Read_OutFmap_Bfloat2Dec(OutImage8_1st_Layer7_CH0, OutImage8_1st_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
    OutImages_1st_Layer7 = OutImage1_1st_Layer7 + OutImage2_1st_Layer7 + OutImage3_1st_Layer7 + OutImage4_1st_Layer7 + \
                            OutImage5_1st_Layer7 + OutImage6_1st_Layer7 + OutImage7_1st_Layer7 + OutImage8_1st_Layer7    
    OutImage_1st_Layer7 = torch.tensor([float(value) for value in OutImages_1st_Layer7], dtype=torch.float32).reshape(8, 1024, 13, 13)
    
    # Calculate Mean/Var 
    Mean_1st_Layer7, Var_1st_Layer7 = Cal_mean_var.forward(OutImage_1st_Layer7)
    Mean_1st_Layer7, Var_1st_Layer7 = Mean_Var_Dec2Bfloat(Mean_1st_Layer7, Var_1st_Layer7, Exponent_Bits, Mantissa_Bits)
    Weight_2nd_Layer7 = New_Weight_Hardware_ReOrdering_OtherLayer(1024, 1024, Weight_List[7], Mean_1st_Layer7, Var_1st_Layer7, Beta_List[7], Iteration="2")
    print("Weight_2nd_Layer7_CH0: " + str(len(Weight_2nd_Layer7[0])))
    print("Weight_2nd_Layer7_CH1: " + str(len(Weight_2nd_Layer7[1])))
    
    # Write Weight_2nd_Layer7 to DDR: 
    Weight_2nd_Layer7_CH0 = Break_FlipHex_256To32(Weight_2nd_Layer7[0])
    Weight_2nd_Layer7_CH1 = Break_FlipHex_256To32(Weight_2nd_Layer7[1])
    
    # Write Weight_2nd_Layer7 to DDR: 
    PCIe2DDR(Weight_2nd_Layer7_CH0, Wr_Address=None)
    print("\t --> " + " Write Weight_2nd_Layer7_CH0 Done!")
    PCIe2DDR(Weight_2nd_Layer7_CH1, Wr_Address=None)
    print("\t --> " + " Write Weight_2nd_Layer7_CH1 Done!")    
    ###########################################2nd Iteration#############################################

    #####################################################################################################
    #                                                                                                   #
    #                                         Layer8_Forward                                            #
    #                                                                                                   #
    #####################################################################################################  
    #################################################Hardware############################################
    toc = time.time()
    cost_time3 = (toc-tic)
    Forward_Time = cost_time3
# Hardware Final Result

print("=== Our Hardware-Processing is finished! ===\n\n=== Let's Starting Post-Processing: ===")

# Post-Processing Pre-Defined Conditions
Post_Start_Signal = None

# OutputImage from Hardware

# Post Processing
if Post_Start_Signal == "1" or Post_Start_Signal == "1".zfill(4) or Post_Start_Signal == "1".zfill(16):
    OutImage1_2nd_Layer8_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage1_2nd_Layer8_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_2nd_Layer8_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage2_2nd_Layer8_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_2nd_Layer8_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage3_2nd_Layer8_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_2nd_Layer8_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage4_2nd_Layer8_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_2nd_Layer8_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage5_2nd_Layer8_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_2nd_Layer8_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage6_2nd_Layer8_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_2nd_Layer8_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage7_2nd_Layer8_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_2nd_Layer8_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    OutImage8_2nd_Layer8_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    PostProcessing = Post_Processing(Mode=Mode,
                            Brain_Floating_Point=Brain_Floating_Point,
                            Exponent_Bits=Exponent_Bits,
                            Mantissa_Bits=Mantissa_Bits,
                            OutImage1_Data_CH0=OutImage1_2nd_Layer8_CH0,
                            OutImage1_Data_CH1=OutImage1_2nd_Layer8_CH1,
                            OutImage2_Data_CH0=OutImage2_2nd_Layer8_CH0,
                            OutImage2_Data_CH1=OutImage2_2nd_Layer8_CH1,
                            OutImage3_Data_CH0=OutImage3_2nd_Layer8_CH0,
                            OutImage3_Data_CH1=OutImage3_2nd_Layer8_CH1,
                            OutImage4_Data_CH0=OutImage4_2nd_Layer8_CH0,
                            OutImage4_Data_CH1=OutImage4_2nd_Layer8_CH1,
                            OutImage5_Data_CH0=OutImage5_2nd_Layer8_CH0,
                            OutImage5_Data_CH1=OutImage5_2nd_Layer8_CH1,
                            OutImage6_Data_CH0=OutImage6_2nd_Layer8_CH0,
                            OutImage6_Data_CH1=OutImage6_2nd_Layer8_CH1,
                            OutImage7_Data_CH0=OutImage7_2nd_Layer8_CH0,
                            OutImage7_Data_CH1=OutImage7_2nd_Layer8_CH1,
                            OutImage8_Data_CH0=OutImage8_2nd_Layer8_CH0,
                            OutImage8_Data_CH1=OutImage8_2nd_Layer8_CH1
                            )
    if Mode == "Training":
        tic = time.time()
        Loss, Loss_Gradient = PostProcessing.PostProcessing()
        toc = time.time()
        cost_time13 = (toc-tic)
        Loss_Calculation_Time = cost_time13
        print(Loss)
        print(Loss_Gradient)
        # print(f"Loss Calculation Time: {(Loss_Calculation_Time):.2f} s\n")
    if Mode == "Inference":
        tic = time.time()
        Loss, _ = PostProcessing.PostProcessing()
        toc = time.time()
        cost_time13 = (toc-tic)
        print(Loss)
        Inference_Time = Pre_Processing_Time + Forward_Time + Loss_Calculation_Time
        print("\n")
        print(tabulate([['Pre-Processing', Pre_Processing_Time], ['YOLOv2-Tiny Forward', Forward_Time],
                    ['Loss Calculation', Loss_Calculation_Time],
                    ['--------------------', str('----------------------')],
                    ['Total Spending Time', Inference_Time]], 
                    headers=['Python Training Process Type', 'Total Time (s)'], tablefmt='orgtbl'))
        print("\n")
        # Detection and Boundary Boxes

print("Our Training Forward Propagation is finished!")

# Hardware Backpropagation Initialization
if YOLOv2_Hardware_Forward:
    if Mode == "Training": 
        tic = time.time()
        toc = time.time()
        cost_time14 = (toc-tic)
        Backward_Time = 0
        None

    Training_Time = Pre_Processing_Time + Forward_Time + Loss_Calculation_Time + Backward_Time
    print(f"Total YOLOv2-Tiny Time for 9Layers with 8Images: {(Training_Time):.2f} s\n")
    print("\n")
    print(tabulate([['Pre-Processing', Pre_Processing_Time], ['YOLOv2-Tiny Forward', Forward_Time],
                    ['Loss Calculation', Loss_Calculation_Time], ['YOLOv2-Tiny Backward', Backward_Time], 
                    ['--------------------', str('----------------------')],
                    ['Total Spending Time in (s)', Training_Time],
                    ['Total Spending Time in (mn)', Training_Time/60]], 
                    headers=['Python Training Process Type', 'Total Time (s)'], tablefmt='orgtbl'))
    print("\n")

# Backpropagation
######################################################################################################
#                                                                                                    #
#                                         Pre-Processing Backward                                    #
#                                                                                                    #
######################################################################################################
if BN_Param_Converted:
    tic = time.time()
    Backward_Const_List = PreProcessing.Backward_Const_Param_Converted_Func()
    toc = time.time()
    cost_time15 = (toc-tic)
    
    tic = time.time()
    Average_Per_Channel_List = PreProcessing.Average_Per_Channel_Param_Converted_Func()
    toc = time.time()
    cost_time16 = (toc-tic)

if YOLOv2_Hardware_Backward:
    # Weight_Backward_Layer8 for Soft2Hardware
    Weight_Backward_Layer8 = Weight_Hardware_Backward_ReOrdering_OtherLayer(128, 1024, Weight_List[8], ["0000"]*125, ["0000"]*125)
    print("Weight_Backward_Layer8: " + str(len(Weight_Backward_Layer8[0])))
    print("Weight_Backward_Layer8: " + str(len(Weight_Backward_Layer8[1])))
    
    # Weight_Backward_Layer7 for Soft2Hardware
    Weight_Backward_Layer7 = Weight_Hardware_Backward_ReOrdering_OtherLayer(1024, 1024, Weight_List[7], Backward_Const_List[7], Average_Per_Channel_List[7])
    print("Weight_Backward_Layer7: " + str(len(Weight_Backward_Layer7[0])))
    print("Weight_Backward_Layer7: " + str(len(Weight_Backward_Layer7[1])))
    
    # Weight_Backward_Layer6 for Soft2Hardware
    Weight_Backward_Layer6 = Weight_Hardware_Backward_ReOrdering_OtherLayer(1024, 512, Weight_List[6], Backward_Const_List[6], Average_Per_Channel_List[6])
    print("Weight_Backward_Layer6: " + str(len(Weight_Backward_Layer6[0])))
    print("Weight_Backward_Layer6: " + str(len(Weight_Backward_Layer6[1])))

    # Weight_Backward_Layer5 for Soft2Hardware
    Weight_Backward_Layer5 = Weight_Hardware_Backward_ReOrdering_OtherLayer(512, 256, Weight_List[5], Backward_Const_List[5], Average_Per_Channel_List[5])
    print("Weight_Backward_Layer5: " + str(len(Weight_Backward_Layer5[0])))
    print("Weight_Backward_Layer5: " + str(len(Weight_Backward_Layer5[1])))

    # Weight_Backward_Layer4 for Soft2Hardware
    Weight_Backward_Layer4 = Weight_Hardware_Backward_ReOrdering_OtherLayer(256, 128, Weight_List[4], Backward_Const_List[4], Average_Per_Channel_List[4])
    print("Weight_Backward_Layer4: " + str(len(Weight_Backward_Layer4[0])))
    print("Weight_Backward_Layer4: " + str(len(Weight_Backward_Layer4[1])))

    # Weight_Backward_Layer3 for Soft2Hardware
    Weight_Backward_Layer3 = Weight_Hardware_Backward_ReOrdering_OtherLayer(128, 64, Weight_List[3], Backward_Const_List[3], Average_Per_Channel_List[3])
    print("Weight_Backward_Layer3: " + str(len(Weight_Backward_Layer3[0])))
    print("Weight_Backward_Layer3: " + str(len(Weight_Backward_Layer3[1])))

    # Weight_Backward_Layer2 for Soft2Hardware
    Weight_Backward_Layer2 = Weight_Hardware_Backward_ReOrdering_OtherLayer(64, 32, Weight_List[2], Backward_Const_List[2], Average_Per_Channel_List[2])
    print("Weight_Backward_Layer2: " + str(len(Weight_Backward_Layer2[0])))
    print("Weight_Backward_Layer2: " + str(len(Weight_Backward_Layer2[1])))

    # Weight_Backward_Layer1 for Soft2Hardware
    Weight_Backward_Layer1 = Weight_Hardware_Backward_ReOrdering_OtherLayer(32, 16, Weight_List[1], Backward_Const_List[1], Average_Per_Channel_List[1])
    print("Weight_Backward_Layer1: " + str(len(Weight_Backward_Layer1[0])))
    print("Weight_Backward_Layer1: " + str(len(Weight_Backward_Layer1[1])))

    # Weight for Soft2Hardware
    Weight_Backward_Layer0 = Weight_Hardware_Backward_ReOrdering_Layer0(16, 16, Weight_List[0], Backward_Const_List[0], Average_Per_Channel_List[0])
    print("Weight_Layer0: " + str(len(Weight_Backward_Layer0[0])))
    print("Weight_Layer0: " + str(len(Weight_Backward_Layer0[1])))
    
    # Total Weight for Backward: 
    Weight_Backward_CH0 = Weight_Backward_Layer0[0] + Weight_Backward_Layer1[0] + Weight_Backward_Layer2[0] + Weight_Backward_Layer3[0] + Weight_Backward_Layer4[0] + \
                    Weight_Backward_Layer5[0] + Weight_1st_Layer6[0] + Weight_Backward_Layer7[0] + Weight_Backward_Layer8[0]
    Weight_Backward_CH1 = Weight_Backward_Layer0[1] + Weight_Backward_Layer6[1] + Weight_Backward_Layer2[1] + Weight_Backward_Layer3[1] + Weight_Backward_Layer4[1] + \
                    Weight_Backward_Layer5[1] + Weight_Backward_Layer6[1] + Weight_Backward_Layer7[1] + Weight_Backward_Layer8[1]
    
    # Break 256To32 and Flip the Data: 
    Weight_Backward_CH0 = Break_FlipHex_256To32(Weight_Backward_CH0)
    Weight_Backward_CH1 = Break_FlipHex_256To32(Weight_Backward_CH1)
    
    # Write Weight For Backward into DDR
    PCIe2DDR(Weight_Backward_CH0, Wr_Address=None)
    print("\t --> " + " Write Weight_Backward_CH0 Done!")
    PCIe2DDR(Weight_Backward_CH1, Wr_Address=None)
    print("\t --> " + " Write Weight_Backward_CH1 Done!")
    
    # Loss Gradient for Soft2Hardware
    Loss_Gradient1_layer8 = Loss_Gradient[0:1]  
    Loss_Gradient2_layer8 = Loss_Gradient[1:2]  
    Loss_Gradient3_layer8 = Loss_Gradient[2:3]  
    Loss_Gradient4_layer8 = Loss_Gradient[3:4]
    Loss_Gradient5_layer8 = Loss_Gradient[4:5]  
    Loss_Gradient6_layer8 = Loss_Gradient[5:6]  
    Loss_Gradient7_layer8 = Loss_Gradient[6:7]  
    Loss_Gradient8_layer8 = Loss_Gradient[7:8]
    # Loss_Grad1:
    Loss_Grad1_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient1_layer8, Exponent_Bits, Mantissa_Bits)
    Loss_Grad1_layer8 = Fmap_Hardware_ReOrdering(Out_Channel=128, Data_List=Loss_Grad1_layer8)
    # Loss_Grad2:
    Loss_Grad2_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient2_layer8, Exponent_Bits, Mantissa_Bits)
    Loss_Grad2_layer8 = Fmap_Hardware_ReOrdering(Out_Channel=128, Data_List=Loss_Grad2_layer8) 
    # Loss_Grad3:  
    Loss_Grad3_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient3_layer8, Exponent_Bits, Mantissa_Bits)
    Loss_Grad3_layer8 = Fmap_Hardware_ReOrdering(Out_Channel=128, Data_List=Loss_Grad3_layer8) 
    # Loss_Grad4:  
    Loss_Grad4_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient4_layer8, Exponent_Bits, Mantissa_Bits)
    Loss_Grad4_layer8 = Fmap_Hardware_ReOrdering(Out_Channel=128, Data_List=Loss_Grad4_layer8) 
    # Loss_Grad5:
    Loss_Grad5_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient5_layer8, Exponent_Bits, Mantissa_Bits)
    Loss_Grad5_layer8 = Fmap_Hardware_ReOrdering(Out_Channel=128, Data_List=Loss_Grad5_layer8) 
    # Loss_Grad6:
    Loss_Grad6_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient6_layer8, Exponent_Bits, Mantissa_Bits)
    Loss_Grad6_layer8 = Fmap_Hardware_ReOrdering(Out_Channel=128, Data_List=Loss_Grad6_layer8) 
    # Loss_Grad7:  
    Loss_Grad7_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient7_layer8, Exponent_Bits, Mantissa_Bits)
    Loss_Grad7_layer8 = Fmap_Hardware_ReOrdering(Out_Channel=128, Data_List=Loss_Grad7_layer8) 
    # Loss_Grad8:  
    Loss_Grad8_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient8_layer8, Exponent_Bits, Mantissa_Bits)
    Loss_Grad8_layer8 = Fmap_Hardware_ReOrdering(Out_Channel=128, Data_List=Loss_Grad8_layer8)
    
    # Separate the DDR Channel: 
    Loss_Grad_layer8_CH0 =  Loss_Grad1_layer8[0] + Loss_Grad2_layer8[0] + Loss_Grad3_layer8[0] + Loss_Grad4_layer8[0] + \
                            Loss_Grad5_layer8[0] + Loss_Grad6_layer8[0] + Loss_Grad7_layer8[0] + Loss_Grad8_layer8[0]
    
    Loss_Grad_layer8_CH1 =  Loss_Grad1_layer8[1] + Loss_Grad2_layer8[1] + Loss_Grad3_layer8[1] + Loss_Grad4_layer8[1] + \
                            Loss_Grad5_layer8[1] + Loss_Grad6_layer8[1] + Loss_Grad7_layer8[1] + Loss_Grad8_layer8[1]
    
    # Write Loss Gradient to DDR:
    PCIe2DDR(Loss_Grad_layer8_CH0, Wr_Address=None)
    print("\t --> " + " Write Loss_Grad_layer8_CH0 Done!")
    PCIe2DDR(Loss_Grad_layer8_CH1, Wr_Address=None)
    print("\t --> " + " Write Loss_Grad_layer8_CH1 Done!") 

######################################################################################################
#                                                                                                    #
#                                         Layer8_Backward                                            #
#                                                                                                    #
######################################################################################################       
    # Weight Gradient
    Weight_Gradient1_Layer8_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient1_Layer8_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer8_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer8_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer8_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer8_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer8_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer8_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer8_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer8_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer8_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer8_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer8_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer8_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer8_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer8_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    
    Weight_Gradient1_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer8_CH0, Weight_Gradient1_Layer8_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
    Weight_Gradient2_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer8_CH0, Weight_Gradient2_Layer8_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
    Weight_Gradient3_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer8_CH0, Weight_Gradient3_Layer8_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
    Weight_Gradient4_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer8_CH0, Weight_Gradient4_Layer8_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
    Weight_Gradient5_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer8_CH0, Weight_Gradient5_Layer8_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
    Weight_Gradient6_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer8_CH0, Weight_Gradient6_Layer8_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
    Weight_Gradient7_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer8_CH0, Weight_Gradient7_Layer8_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
    Weight_Gradient8_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer8_CH0, Weight_Gradient8_Layer8_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
    Weight_Gradient_Layer8 = [Weight_Gradient1_Layer8, Weight_Gradient2_Layer8, Weight_Gradient3_Layer8, Weight_Gradient4_Layer8, Weight_Gradient5_Layer8, 
                              Weight_Gradient6_Layer8, Weight_Gradient7_Layer8, Weight_Gradient8_Layer8]
    Weight_Gradient_Layer8 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer8)]   
    Weight_Gradient_Layer8 = torch.tensor([float(value) for value in Weight_Gradient_Layer8], dtype=torch.float32).reshape(128, 1024, 1, 1)

######################################################################################################
#                                                                                                    #
#                                         Layer7_Backward                                            #
#                                                                                                    #
######################################################################################################

    # Weight Gradient
    Weight_Gradient1_Layer7_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient1_Layer7_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer7_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer7_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer7_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer7_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer7_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer7_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer7_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer7_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer7_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer7_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer7_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer7_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer7_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer7_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    
    Weight_Gradient1_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer7_CH0, Weight_Gradient1_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
    Weight_Gradient2_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer7_CH0, Weight_Gradient2_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
    Weight_Gradient3_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer7_CH0, Weight_Gradient3_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
    Weight_Gradient4_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer7_CH0, Weight_Gradient4_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
    Weight_Gradient5_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer7_CH0, Weight_Gradient5_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
    Weight_Gradient6_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer7_CH0, Weight_Gradient6_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
    Weight_Gradient7_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer7_CH0, Weight_Gradient7_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
    Weight_Gradient8_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer7_CH0, Weight_Gradient8_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
    Weight_Gradient_Layer7 = [Weight_Gradient1_Layer7, Weight_Gradient2_Layer7, Weight_Gradient3_Layer7, Weight_Gradient4_Layer7, Weight_Gradient5_Layer7, 
                              Weight_Gradient6_Layer7, Weight_Gradient7_Layer7, Weight_Gradient8_Layer7]
    Weight_Gradient_Layer7 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer7)]   
    Weight_Gradient_Layer7 = torch.tensor([float(value) for value in Weight_Gradient_Layer7], dtype=torch.float32).reshape(1024, 1024, 3, 3)

######################################################################################################
#                                                                                                    #
#                                         Layer6_Backward                                            #
#                                                                                                    #
######################################################################################################
    
    # Weight Gradient
    Weight_Gradient1_Layer6_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient1_Layer6_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer6_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer6_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer6_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer6_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer6_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer6_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer6_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer6_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer6_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer6_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer6_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer6_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer6_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer6_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    
    Weight_Gradient1_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer6_CH0, Weight_Gradient1_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
    Weight_Gradient2_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer6_CH0, Weight_Gradient2_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
    Weight_Gradient3_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer6_CH0, Weight_Gradient3_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
    Weight_Gradient4_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer6_CH0, Weight_Gradient4_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
    Weight_Gradient5_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer6_CH0, Weight_Gradient5_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
    Weight_Gradient6_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer6_CH0, Weight_Gradient6_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
    Weight_Gradient7_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer6_CH0, Weight_Gradient7_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
    Weight_Gradient8_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer6_CH0, Weight_Gradient8_Layer6_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
    Weight_Gradient_Layer6 = [Weight_Gradient1_Layer6, Weight_Gradient2_Layer6, Weight_Gradient3_Layer6, Weight_Gradient4_Layer6, Weight_Gradient5_Layer6, 
                              Weight_Gradient6_Layer6, Weight_Gradient7_Layer6, Weight_Gradient8_Layer6]
    Weight_Gradient_Layer6 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer6)]   
    Weight_Gradient_Layer6 = torch.tensor([float(value) for value in Weight_Gradient_Layer6], dtype=torch.float32).reshape(1024, 512, 3, 3)

######################################################################################################
#                                                                                                    #
#                                         Layer5_Backward                                            #
#                                                                                                    #
######################################################################################################
    
    # Weight Gradient
    Weight_Gradient1_Layer5_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient1_Layer5_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer5_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer5_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer5_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer5_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer5_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer5_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer5_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer5_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer5_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer5_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer5_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer5_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer5_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer5_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    
    Weight_Gradient1_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer5_CH0, Weight_Gradient1_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
    Weight_Gradient2_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer5_CH0, Weight_Gradient2_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
    Weight_Gradient3_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer5_CH0, Weight_Gradient3_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
    Weight_Gradient4_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer5_CH0, Weight_Gradient4_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
    Weight_Gradient5_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer5_CH0, Weight_Gradient5_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
    Weight_Gradient6_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer5_CH0, Weight_Gradient6_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
    Weight_Gradient7_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer5_CH0, Weight_Gradient7_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
    Weight_Gradient8_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer5_CH0, Weight_Gradient8_Layer5_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
    Weight_Gradient_Layer5 = [Weight_Gradient1_Layer5, Weight_Gradient2_Layer5, Weight_Gradient3_Layer5, Weight_Gradient4_Layer5, Weight_Gradient5_Layer5, 
                              Weight_Gradient6_Layer5, Weight_Gradient7_Layer5, Weight_Gradient8_Layer5]
    Weight_Gradient_Layer5 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer5)]   
    Weight_Gradient_Layer5 = torch.tensor([float(value) for value in Weight_Gradient_Layer5], dtype=torch.float32).reshape(512, 256, 3, 3)

######################################################################################################
#                                                                                                    #
#                                         Layer4_Backward                                            #
#                                                                                                    #
######################################################################################################

    # Weight Gradient
    Weight_Gradient1_Layer4_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient1_Layer4_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer4_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer4_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer4_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer4_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer4_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer4_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer4_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer4_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer4_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer4_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer4_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer4_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer4_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer4_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    
    Weight_Gradient1_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer4_CH0, Weight_Gradient1_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
    Weight_Gradient2_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer4_CH0, Weight_Gradient2_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
    Weight_Gradient3_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer4_CH0, Weight_Gradient3_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
    Weight_Gradient4_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer4_CH0, Weight_Gradient4_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
    Weight_Gradient5_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer4_CH0, Weight_Gradient5_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
    Weight_Gradient6_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer4_CH0, Weight_Gradient6_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
    Weight_Gradient7_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer4_CH0, Weight_Gradient7_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
    Weight_Gradient8_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer4_CH0, Weight_Gradient8_Layer4_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
    Weight_Gradient_Layer4 = [Weight_Gradient1_Layer4, Weight_Gradient2_Layer4, Weight_Gradient3_Layer4, Weight_Gradient4_Layer4, Weight_Gradient5_Layer4, 
                              Weight_Gradient6_Layer4, Weight_Gradient7_Layer4, Weight_Gradient8_Layer4]
    Weight_Gradient_Layer4 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer4)]   
    Weight_Gradient_Layer4 = torch.tensor([float(value) for value in Weight_Gradient_Layer4], dtype=torch.float32).reshape(256, 128, 3, 3)

######################################################################################################
#                                                                                                    #
#                                         Layer3_Backward                                            #
#                                                                                                    #
######################################################################################################
    
    # Weight Gradient
    Weight_Gradient1_Layer3_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient1_Layer3_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer3_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer3_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer3_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer3_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer3_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer3_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer3_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer3_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer3_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer3_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer3_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer3_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer3_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer3_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    
    Weight_Gradient1_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer3_CH0, Weight_Gradient1_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
    Weight_Gradient2_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer3_CH0, Weight_Gradient2_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
    Weight_Gradient3_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer3_CH0, Weight_Gradient3_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
    Weight_Gradient4_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer3_CH0, Weight_Gradient4_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
    Weight_Gradient5_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer3_CH0, Weight_Gradient5_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
    Weight_Gradient6_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer3_CH0, Weight_Gradient6_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
    Weight_Gradient7_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer3_CH0, Weight_Gradient7_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
    Weight_Gradient8_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer3_CH0, Weight_Gradient8_Layer3_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
    Weight_Gradient_Layer3 = [Weight_Gradient1_Layer3, Weight_Gradient2_Layer3, Weight_Gradient3_Layer3, Weight_Gradient4_Layer3, Weight_Gradient5_Layer3, 
                              Weight_Gradient6_Layer3, Weight_Gradient7_Layer3, Weight_Gradient8_Layer3]
    Weight_Gradient_Layer3 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer3)]   
    Weight_Gradient_Layer3 = torch.tensor([float(value) for value in Weight_Gradient_Layer3], dtype=torch.float32).reshape(128, 64, 3, 3)

######################################################################################################
#                                                                                                    #
#                                         Layer2_Backward                                            #
#                                                                                                    #
######################################################################################################
    
    # Weight Gradient
    Weight_Gradient1_Layer2_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient1_Layer2_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer2_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer2_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer2_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer2_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer2_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer2_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer2_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer2_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer2_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer2_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer2_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer2_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer2_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer2_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    
    Weight_Gradient1_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer2_CH0, Weight_Gradient1_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
    Weight_Gradient2_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer2_CH0, Weight_Gradient2_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
    Weight_Gradient3_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer2_CH0, Weight_Gradient3_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
    Weight_Gradient4_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer2_CH0, Weight_Gradient4_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
    Weight_Gradient5_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer2_CH0, Weight_Gradient5_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
    Weight_Gradient6_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer2_CH0, Weight_Gradient6_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
    Weight_Gradient7_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer2_CH0, Weight_Gradient7_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
    Weight_Gradient8_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer2_CH0, Weight_Gradient8_Layer2_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
    Weight_Gradient_Layer2 = [Weight_Gradient1_Layer2, Weight_Gradient2_Layer2, Weight_Gradient3_Layer2, Weight_Gradient4_Layer2, Weight_Gradient5_Layer2, 
                              Weight_Gradient6_Layer2, Weight_Gradient7_Layer2, Weight_Gradient8_Layer2]
    Weight_Gradient_Layer2 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer2)]   
    Weight_Gradient_Layer2 = torch.tensor([float(value) for value in Weight_Gradient_Layer2], dtype=torch.float32).reshape(64, 32, 3, 3)

######################################################################################################
#                                                                                                    #
#                                         Layer1_Backward                                            #
#                                                                                                    #
######################################################################################################
    
    # Weight Gradient
    Weight_Gradient1_Layer1_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient1_Layer1_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer1_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer1_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer1_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer1_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer1_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer1_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer1_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer1_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer1_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer1_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer1_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer1_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer1_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer1_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    
    Weight_Gradient1_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer1_CH0, Weight_Gradient1_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
    Weight_Gradient2_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer1_CH0, Weight_Gradient2_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
    Weight_Gradient3_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer1_CH0, Weight_Gradient3_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
    Weight_Gradient4_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer1_CH0, Weight_Gradient4_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
    Weight_Gradient5_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer1_CH0, Weight_Gradient5_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
    Weight_Gradient6_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer1_CH0, Weight_Gradient6_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
    Weight_Gradient7_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer1_CH0, Weight_Gradient7_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
    Weight_Gradient8_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer1_CH0, Weight_Gradient8_Layer1_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
    Weight_Gradient_Layer1 = [Weight_Gradient1_Layer1, Weight_Gradient2_Layer1, Weight_Gradient3_Layer1, Weight_Gradient4_Layer1, Weight_Gradient5_Layer1, 
                              Weight_Gradient6_Layer1, Weight_Gradient7_Layer1, Weight_Gradient8_Layer1]
    Weight_Gradient_Layer1 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer1)]   
    Weight_Gradient_Layer1 = torch.tensor([float(value) for value in Weight_Gradient_Layer1], dtype=torch.float32).reshape(32, 16, 3, 3)

######################################################################################################
#                                                                                                    #
#                                         Layer0_Backward                                            #
#                                                                                                    #
######################################################################################################
    
    # Weight Gradient
    Weight_Gradient1_Layer0_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient1_Layer0_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer0_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient2_Layer0_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer0_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient3_Layer0_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer0_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient4_Layer0_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer0_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient5_Layer0_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer0_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient6_Layer0_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer0_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient7_Layer0_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer0_CH0 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    Weight_Gradient8_Layer0_CH1 = DDR2PCIe(Rd_Address=None, Start_Offset=None, End_Offset=None)
    
    Weight_Gradient1_Layer0 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer0_CH0, Weight_Gradient1_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
    Weight_Gradient2_Layer0 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer0_CH0, Weight_Gradient2_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
    Weight_Gradient3_Layer0 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer0_CH0, Weight_Gradient3_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
    Weight_Gradient4_Layer0 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer0_CH0, Weight_Gradient4_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
    Weight_Gradient5_Layer0 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer0_CH0, Weight_Gradient5_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
    Weight_Gradient6_Layer0 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer0_CH0, Weight_Gradient6_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
    Weight_Gradient7_Layer0 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer0_CH0, Weight_Gradient7_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
    Weight_Gradient8_Layer0 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer0_CH0, Weight_Gradient8_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
    Weight_Gradient_Layer0 = [Weight_Gradient1_Layer0, Weight_Gradient2_Layer0, Weight_Gradient3_Layer0, Weight_Gradient4_Layer0, Weight_Gradient5_Layer0, 
                              Weight_Gradient6_Layer0, Weight_Gradient7_Layer0, Weight_Gradient8_Layer0]
    Weight_Gradient_Layer0 = [sum(item) / len(item) for item in zip(*Weight_Gradient_Layer0)]   
    Weight_Gradient_Layer0 = torch.tensor([float(value) for value in Weight_Gradient_Layer0], dtype=torch.float32).reshape(32, 16, 3, 3)
    
######################################################################################################
#                                                                                                    #
#                                         Weight Updating                                            #
#                                                                                                    #
######################################################################################################