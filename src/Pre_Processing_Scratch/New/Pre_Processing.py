from .Pre_Processing_Function import *


# from Pre_Processing_Function import *


# 2023/06/20: Implemented By Thaising
# Combined Master-PhD in MSISLAB

class Pre_Processing:
    def __init__(self, Mode, Brain_Floating_Point, Exponent_Bits, Mantissa_Bits):
        self.Mode = Mode
        self.Brain_Floating_Point = Brain_Floating_Point
        self.Exponent_Bits = Exponent_Bits
        self.Mantissa_Bits = Mantissa_Bits

        # Image Width and Height
        self.frameWidth = 416
        self.frameHeight = 416

    def Image_Converted_Func(self, Image_Read_Path):
        # Mode is Training
        global Image
        if self.Mode == "Training":
            # Loading Image & Write the Image

            # Using VOC_Dataset_Image
            if self.Brain_Floating_Point:
                Image = Read_Image_into_BFP(Image_Read_Path, self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Image = Read_Image_into_FP32(Image_Read_Path, self.Exponent_Bits, self.Mantissa_Bits)
        # Mode is Inference
        if self.Mode == "Inference":
            # Using Random Selected Image
            Image_Directory()
            Image_Path = '../Pre_Processing_Scratch/Images/Photo.jpeg'
            image = Load_Image(Image_Path, self.frameWidth, self.frameHeight)
            Image_Write_Path = "../Main_Processing_Scratch/Pre_Image_Converted/Image.mem"
            Image_Read_Path = "../Main_Processing_Scratch/Pre_Image_Converted/Image.mem"
            if self.Brain_Floating_Point:
                Write_Image_into_BFP(Image_Write_Path, image, self.Exponent_Bits, self.Mantissa_Bits)
                Image = Read_Image(Image_Read_Path)
            else:
                Write_Image_into_FP32(Image_Write_Path, image, self.Exponent_Bits, self.Mantissa_Bits)
                Image = Read_Image(Image_Read_Path)
        return Image

    def Weight_Converted_Func(self):
        # Mode is Training
        global Weight_List
        if self.Mode == "Training":
            # Loading Conv_Weight and Write Conv_Weight
            Read_Weight_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Conv_Param"
            File_List_Weight = os.listdir(Read_Weight_Folder_Path)
            if self.Brain_Floating_Point:
                Weight_List = Read_Weight_into_BFP(File_List_Weight, Read_Weight_Folder_Path,
                                                   self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Weight_List = Read_Weight_into_FP32(File_List_Weight, Read_Weight_Folder_Path,
                                                    self.Exponent_Bits, self.Mantissa_Bits)
        if self.Mode == "Inference":
            # Loading Conv_Weight and Write Conv_Weight
            print("\t --> " + " The Weights are processing, Please Wait for a Moment!")
            Read_Weight_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Conv_Param"
            File_List_Weight = os.listdir(Read_Weight_Folder_Path)
            if self.Brain_Floating_Point:
                Weight_List = Read_Weight_into_BFP(File_List_Weight, Read_Weight_Folder_Path,
                                                   self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Weight_List = Read_Weight_into_FP32(File_List_Weight, Read_Weight_Folder_Path,
                                                    self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The Weights are Successfully Converted!")
        return Weight_List

    def Bias_Converted_Func(self):
        # Mode is Training
        global Bias
        if self.Mode == "Training":
            # Loading Bias and Write Bias:
            print("\t --> " + " The Bias is processing, Please Wait for a Moment!")
            Read_Bias_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Bias"
            File_List_Bias = os.listdir(Read_Bias_Folder_Path)
            if self.Brain_Floating_Point:
                Bias = Read_Bias_into_BFP(File_List_Bias, Read_Bias_Folder_Path,
                                   self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Bias = Read_Bias_into_FP32(File_List_Bias, Read_Bias_Folder_Path,
                                     self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The Bias is Successfully Converted!")

        # Mode is Inference
        if self.Mode == "Inference":
            # Loading Bias and Write Bias:
            print("\t --> " + " The Bias is processing, Please Wait for a Moment!")
            Read_Bias_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/Bias"
            File_List_Bias = os.listdir(Read_Bias_Folder_Path)
            if self.Brain_Floating_Point:
                Bias = Read_Bias_into_BFP(File_List_Bias, Read_Bias_Folder_Path,
                                          self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Bias = Read_Bias_into_FP32(File_List_Bias, Read_Bias_Folder_Path,
                                           self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The Bias is Successfully Converted!")
        return Bias

    def Beta_Param_Converted_Func(self):
        # Mode is Training
        global Beta_List
        if self.Mode == "Training":
            # Loading Beta and Write Beta:
            print("\t --> " + " The BN Param: Beta is processing, Please Wait for a Moment!")
            Read_Beta_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Beta"
            File_List_Beta = os.listdir(Read_Beta_Folder_Path)
            if self.Brain_Floating_Point:
                Beta_List = Read_BN_Parameters_into_BFP(File_List_Beta, Read_Beta_Folder_Path,
                                            self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Beta_List = Read_BN_Parameters_into_FP32(File_List_Beta, Read_Beta_Folder_Path,
                                             self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The BN Param: Beta is Successfully Converted!")

        # Mode is Inference
        if self.Mode == "Inference":
            # Loading Beta and Write Beta:
            print("\t --> " + " The BN Param: Beta is processing, Please Wait for a Moment!")
            Read_Beta_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Beta"
            File_List_Beta = os.listdir(Read_Beta_Folder_Path)
            if self.Brain_Floating_Point:
                Beta_List = Read_BN_Parameters_into_BFP(File_List_Beta, Read_Beta_Folder_Path,
                                                        self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Beta_List = Read_BN_Parameters_into_FP32(File_List_Beta, Read_Beta_Folder_Path,
                                                         self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The BN Param: Beta is Successfully Converted!")
        return Beta_List

    def Gamma_Param_Converted_Func(self):
        # Mode is Training
        global Gamma_List
        if self.Mode == "Training":
            # Loading Gamma and Write Gamma:
            print("\t --> " + " The BN Param: Gamma is processing, Please Wait for a Moment!")
            Read_Gamma_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Gamma"
            File_List_Gamma = os.listdir(Read_Gamma_Folder_Path)
            if self.Brain_Floating_Point:
                Gamma_List = Read_BN_Parameters_into_BFP(File_List_Gamma, Read_Gamma_Folder_Path,
                                            self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Gamma_List = Read_BN_Parameters_into_FP32(File_List_Gamma, Read_Gamma_Folder_Path,
                                             self.Exponent_Bits, self.Mantissa_Bits)

            print("\t --> " + " The BN Param: Gamma is Successfully Converted!")

        # Mode is Inference
        if self.Mode == "Inference":
            # Loading Gamma and Write Gamma:
            print("\t --> " + " The BN Param: Gamma is processing, Please Wait for a Moment!")
            Read_Gamma_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Gamma"
            File_List_Gamma = os.listdir(Read_Gamma_Folder_Path)
            if self.Brain_Floating_Point:
                Gamma_List = Read_BN_Parameters_into_BFP(File_List_Gamma, Read_Gamma_Folder_Path,
                                                         self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Gamma_List = Read_BN_Parameters_into_FP32(File_List_Gamma, Read_Gamma_Folder_Path,
                                                          self.Exponent_Bits, self.Mantissa_Bits)

            print("\t --> " + " The BN Param: Gamma is Successfully Converted!")

        return Gamma_List

    def Running_Mean_Param_Converted_Func(self):
        # Mode is Training
        global Running_Mean_List
        if self.Mode == "Training":
            # Loading Running_Mean and Write Running_Mean:
            print("\t --> " + " The BN Param: Running_Mean is processing, Please Wait for a Moment!")
            Read_RunningMean_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Running_Mean"
            File_List_RunningMean = os.listdir(Read_RunningMean_Folder_Path)
            if self.Brain_Floating_Point:
                Running_Mean_List = Read_BN_Parameters_into_BFP(File_List_RunningMean, Read_RunningMean_Folder_Path,
                                            self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Running_Mean_List = Read_BN_Parameters_into_FP32(File_List_RunningMean, Read_RunningMean_Folder_Path,
                                             self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The BN Param: Running_Mean is Successfully Converted!")
        if self.Mode == "Inference":
            # Loading Running_Mean and Write Running_Mean:
            print("\t --> " + " The BN Param: Running_Mean is processing, Please Wait for a Moment!")
            Read_RunningMean_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Running_Mean"
            File_List_RunningMean = os.listdir(Read_RunningMean_Folder_Path)
            if self.Brain_Floating_Point:
                Running_Mean_List = Read_BN_Parameters_into_BFP(File_List_RunningMean, Read_RunningMean_Folder_Path,
                                                                self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Running_Mean_List = Read_BN_Parameters_into_FP32(File_List_RunningMean, Read_RunningMean_Folder_Path,
                                                                 self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The BN Param: Running_Mean is Successfully Converted!")

        return Running_Mean_List

    def Running_Var_Param_Converted_Func(self):
        # Mode is Training
        global Running_Var_List
        if self.Mode == "Training":
            # Loading Running_Var and Write Running_Var:
            print("\t --> " + " The BN Param: Running_Var is processing, Please Wait for a Moment!")
            Read_RunningVar_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Running_Var"
            File_List_RunningVar = os.listdir(Read_RunningVar_Folder_Path)
            if self.Brain_Floating_Point:
                Running_Var_List = Read_BN_Parameters_into_BFP(File_List_RunningVar, Read_RunningVar_Folder_Path,
                                            self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Running_Var_List = Read_BN_Parameters_into_FP32(File_List_RunningVar, Read_RunningVar_Folder_Path,
                                             self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The BN Param: Running_Var is Successfully Converted!")

        # Mode is Inference
        if self.Mode == "Inference":
            # Loading Running_Var and Write Running_Var:
            print("\t --> " + " The BN Param: Running_Var is processing, Please Wait for a Moment!")
            Read_RunningVar_Folder_Path = "/home/msis/Desktop/pcie_python/GUI/Pre_Processing_Scratch/Weight_Parameters/BN_Params/Running_Var"
            File_List_RunningVar = os.listdir(Read_RunningVar_Folder_Path)
            if self.Brain_Floating_Point:
                Running_Var_List = Read_BN_Parameters_into_BFP(File_List_RunningVar, Read_RunningVar_Folder_Path,
                                                               self.Exponent_Bits, self.Mantissa_Bits)
            else:
                Running_Var_List = Read_BN_Parameters_into_FP32(File_List_RunningVar, Read_RunningVar_Folder_Path,
                                                                self.Exponent_Bits, self.Mantissa_Bits)
            print("\t --> " + " The BN Param: Running_Var is Successfully Converted!")

        return Running_Var_List
    
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

def Read_OutFmap_Bfloat2Dec(Data_List_CH0, Data_List_CH1, Exponent_Bit, Mantissa_Bit, Out_CH, Out_Size, Layer8=False):
    # Layer 8: Out_CH=128, Out_Size=13
    # Read all the Weights from a Weight_Folder
    Input_List0 = Data_List_CH0
    Input_List1 = Data_List_CH1
    
    Input_List = Fmap_Reverse_Ordering(Out_CH, Out_Size, Input_List0, Input_List1)
    
    Out_List = []
    if Layer8:
        for Value in Input_List[:(len(Input_List)-3*13*13)]:
            Hexadecimal = str(Value) + "0000"
            Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
            Decimal = Binary2Floating(Binary_Value, Exponent_Bit, Mantissa_Bit)
            Out_List.append(str(Decimal))
    else: 
        for Value in Input_List[:(len(Input_List))]:
            Hexadecimal = str(Value) + "0000"
            Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
            Decimal = Binary2Floating(Binary_Value, Exponent_Bit, Mantissa_Bit)
            Out_List.append(str(Decimal))       
    return Out_List

def origin_idx_calculator(idx, B, H, W, num_chunks):
    origin_idx = []
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

class Cal_mean_var(object):

    @staticmethod
    def Forward(x):
    
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
        scale = 1 / ((avg_max - avg_min) * scale_fix + eps)  

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

def Mean_Var_Dec2Bfloat(Mean, Var, Exponent_Bit, Mantissa_Bit): 
    Mean = Mean.flatten().tolist()
    Var = Var.flatten().tolist()
    Mean_List=[]
    Var_List=[]
    for mean in Mean: 
        Binary_Value = Floating2Binary(mean, Exponent_Bit, Mantissa_Bit)
        Hexadecimal_Value = hex(int(Binary_Value, 2))[2:]
        Truncated_Rounded_Hex = Truncating_Rounding(Hexadecimal_Value)
        Mean_List.append(Truncated_Rounded_Hex)
    for var in Var: 
        Binary_Value = Floating2Binary(var, Exponent_Bit, Mantissa_Bit)
        Hexadecimal_Value = hex(int(Binary_Value, 2))[2:]
        Truncated_Rounded_Hex = Truncating_Rounding(Hexadecimal_Value)
        Var_List.append(Truncated_Rounded_Hex)
    return Mean_List, Var_List

def Loss_Gradient_Dec2Bfloat(Loss_Gradient, Exponent_Bit, Mantissa_Bit): 
    Loss_Gradient = Loss_Gradient.flatten().tolist()
    Loss_Gradient_List=[]
    Zero_List = ["0000"]*(13*13*3)
    for loss_gradient in Loss_Gradient: 
        Binary_Value = Floating2Binary(loss_gradient, Exponent_Bit, Mantissa_Bit)
        Hexadecimal_Value = hex(int(Binary_Value, 2))[2:]
        Truncated_Rounded_Hex = Truncating_Rounding(Hexadecimal_Value)
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
    if Layer8:
        for Value in Input_List[:(len(Input_List)-3*1024)]:
            Hexadecimal = str(Value) + "0000"
            Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
            Decimal = Binary2Floating(Binary_Value, Exponent_Bit, Mantissa_Bit)
            Out_List.append(str(Decimal))
    else: 
        for Value in Input_List[:(len(Input_List))]:
            Hexadecimal = str(Value) + "0000"
            Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
            Decimal = Binary2Floating(Binary_Value, Exponent_Bit, Mantissa_Bit)
            Out_List.append(str(Decimal))       
    return Out_List

def Reversed_FlipHex_32To256(Hex32_List, segment_length=8):
    Reversed_Flip_List = Flip_Data(Hex32_List)
    combined_hex_list = []
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
        scale = 1 / ((avg_max - avg_min) * scale_fix + eps)  

        avg = avg.view(1, -1, 1, 1)
        scale = scale.view(1, -1, 1, 1)

        momentum = 0.1

        output = (x - avg) * scale

        output = output * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)

        cache = (x, gamma, beta, output, scale, scale_fix, avg, avg_max, avg_min, eps, num_chunks, max_index, min_index)
        return cache
