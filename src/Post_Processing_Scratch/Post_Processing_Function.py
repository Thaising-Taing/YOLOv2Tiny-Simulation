import math
import os
import torch
from pathlib import Path
import numpy as np 
import pandas as pd
import struct
import multiprocessing


# 2023/06/21: Implemented By Thaising
# Combined Master-PhD in MSISLAB


def Image_Output_Directory():
    Output = "Post_Image_Output_Converted"
    os.makedirs(Output, exist_ok=True)


def Loss_Directory():
    Output = "../Main_Processing_Scratch/Loss"
    os.makedirs(Output, exist_ok=True)


def Read_and_Write_Image_BFPtoDec(Read_Path, Write_Path, Exponent_Bit, Mantissa_Bit):
    # Read all the Weights from a Weight_Folder
    with open(Read_Path, "r") as file_r:
        Input = file_r.read()

    Input_List = [Value for Value in Input.split()]

    file_w = open(Write_Path, "w+")
    for Value in Input_List:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bit, Mantissa_Bit)
        file_w.write(str(Decimal) + "\n")


def Read_BFPtoDec(Read_Path, Exponent_Bit, Mantissa_Bit):
    # Read all the Weights from a Weight_Folder
    with open(Read_Path, "r") as file_r:
        Input = file_r.read()
    
    Input_List = [Value for Value in Input.split()]
    Out_List = []
    Out_List.clear()
    for Value in Input_List:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bit, Mantissa_Bit)
        Out_List.append(str(Decimal))
    return Out_List

def OutFmap_Layer8_BFPtoDec(List0, List1, Exponent_Bit, Mantissa_Bit):

    Input_List0 = List0
    Input_List1 = List1
    
    Input_List = Fmap_Reverse_Ordering(128, 13, Input_List0, Input_List1)
    
    Out_List = []
    Out_List.clear()
    Out_List=Input_List[:(len(Input_List)-3*13*13)]
         
    return Out_List

def OutFmap_Layer8_FP32toDec(List0, List1, Exponent_Bit, Mantissa_Bit):

    Input_List0 = List0
    Input_List1 = List1
    
    Input_List = Fmap_Reverse_Ordering(128, 13, Input_List0, Input_List1)
    
    Out_List = []
    Out_List.clear()
    for Value in Input_List[:(len(Input_List)-3*13*13)]:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bit, Mantissa_Bit)
        Out_List.append(str(Decimal))
    return Out_List


def Read_FP32toDec(Read_Path, Exponent_Bit, Mantissa_Bit):
    # Read all the Weights from a Weight_Folder
    with open(Read_Path, "r") as file_r:
        Input = file_r.read()

    Input_List = [Value for Value in Input.split()]
    
    Out_List = []
    Out_List.clear()
    for Value in Input_List:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bit, Mantissa_Bit)
        Out_List.append(str(Decimal))
    return Out_List


def Read_and_Write_Image_FP32toDec(Read_Path, Write_Path, Exponent_Bit, Mantissa_Bit):
    # Read all the Weights from a Weight_Folder
    with open(Read_Path, "r") as file_r:
        Input = file_r.read()

    Input_List = [Value for Value in Input.split()]

    file_w = open(Write_Path, "w+")
    for Value in Input_List:
        Hexadecimal = str(Value) + "0000"
        Binary_Value = bin(int(Hexadecimal, 16))[2:].zfill(32)
        Decimal = Binary2Floating(Binary_Value, Exponent_Bit, Mantissa_Bit)
        file_w.write(str(Decimal) + "\n")


def Floating2Binary(num, Exponent_Bit, Mantissa_Bit):
    sign = ('1' if num < 0 else '0')
    num = abs(num)
    bias = math.pow(2, (Exponent_Bit - 1)) - 1
    if num == 0:
        e = 0
    else:
        e = math.floor(math.log(num, 2) + bias)

    if e > (math.pow(2, Exponent_Bit) - 2):  # overflow
        exponent = '1' * Exponent_Bit
        mantissa = '0' * Mantissa_Bit
    else:
        if e > 0:
            s = num / math.pow(2, e - bias) - 1
            exponent = bin(e)[2:].zfill(Exponent_Bit)
        else:  # submoral
            s = num / math.pow(2, (-bias + 1))
            exponent = '0' * Exponent_Bit
        # Rounding Mode By Adding 0.5 (Half-Rounding or Banker's Rounding)
        # Number is smaller or equal 0.5 is rounding down
        # Number is larger 0.5 is rounding up
        mantissa = bin(int(s * (math.pow(2, Mantissa_Bit)) + 0.5))[2:].zfill(Mantissa_Bit)[:Mantissa_Bit]
        # Non-Rounding Mode
        # mantissa = bin(int(s * (math.pow(2, Mantissa_Bit)))[2:].zfill(Mantissa_Bit)[:Mantissa_Bit]
    Floating_Binary = sign + exponent + mantissa

    return Floating_Binary


def Binary2Floating(value, Exponent_Bit, Mantissa_Bit):
    sign = int(value[0], 2)
    if int(value[1:1 + Exponent_Bit], 2) != 0:
        exponent = int(value[1:1 + Exponent_Bit], 2) - int('1' * (Exponent_Bit - 1), 2)
        mantissa = int(value[1 + Exponent_Bit:], 2) * (math.pow(2, (-Mantissa_Bit))) + 1
    else:  # subnormal
        exponent = 1 - int('1' * (Exponent_Bit - 1), 2)
        mantissa = int(value[1 + Exponent_Bit:], 2) * math.pow(2, (-Mantissa_Bit))
    Floating_Decimal = (math.pow(-1, sign)) * (math.pow(2, exponent)) * mantissa
    return Floating_Decimal


def Truncating_Rounding(Truncated_Hexadecimal):
    # Consider the Length of Truncated_Hexadecimal >= 5
    if len(Truncated_Hexadecimal) >= 5:
        # If this Fifth Character Truncated_Hexadecimal >= 5 => Rounding Up the First 16 Bits
        if int(Truncated_Hexadecimal[4], 16) >= 8:
            Rounding_Hexadecimal = hex(int(Truncated_Hexadecimal[:4], 16) + 1)[2:]
        else:
            Rounding_Hexadecimal = Truncated_Hexadecimal[:4]
    else:
        Rounding_Hexadecimal = Truncated_Hexadecimal

    if len(Rounding_Hexadecimal) < 4:
        New_Rounding_Hex = Rounding_Hexadecimal.zfill(4)
    else:
        New_Rounding_Hex = Rounding_Hexadecimal
    Rounding_Hexadecimal_Capitalized = New_Rounding_Hex.upper()
    return Rounding_Hexadecimal_Capitalized


def Loss_into_File(loss, Folder_Path):
    write_path = f'{Folder_Path}/Loss.mem'
    loss_file = open(write_path, mode="w+")
    numerical_loss_value = float(loss.item())
    loss_file.write("Loss : " + str(numerical_loss_value))


def Numerical_Loss(loss):
    numerical_loss_value = float(loss.item())
    # print("Loss : " + str(numerical_loss_value))
    return numerical_loss_value


def Loss_Gradient_into_File(dout_value, Folder_Path, save_hex, save_txt):
    if save_hex:
        write_path = f'{Folder_Path}/Loss_Gradient_hex.mem'
        dout = open(write_path, mode="w+")
        for i in range(dout_value.size(0)):
            for j in range(dout_value.size(1)):
                for k in range(dout_value.size(2)):
                    for l in range(dout_value.size(3)):
                        loss_gradient = float(dout_value[i, j, k, l].item())
                        loss_gradient_bin = Floating2Binary(loss_gradient, 8, 23)
                        loss_gradient_Hex = hex(int(loss_gradient_bin, 2))[2:]
                        loss_gradient_round = Truncating_Rounding(loss_gradient_Hex)
                        dout.write(str(loss_gradient_round) + "\n")
    if save_txt:
        write_path = f'{Folder_Path}/Loss_Gradient.mem'
        dout = open(write_path, mode="w+")
        for i in range(dout_value.size(0)):
            for j in range(dout_value.size(1)):
                for k in range(dout_value.size(2)):
                    for l in range(dout_value.size(3)):
                        loss_gradient = float(dout_value[i, j, k, l].item())
                        dout.write(str(loss_gradient) + "\n")


def save_loss(fname, data, module=[], layer_no=[], save_txt=False, save_hex=False, phase=[]):
    # print(f"Type of data: {type(data)}")
    if save_txt or save_hex:
        if type(data) is dict:
            for _key in data.keys():
                _fname = fname + f'_{_key}'
                save_loss(_fname, data[_key])

        else:
            if module == [] and layer_no == []:
                Out_Path = f'Outputs/{os.path.split(fname)[0]}'
                fname = os.path.split(fname)[1]
            else:
                Out_Path = f'Outputs/Loss/'
                if layer_no != []: Out_Path += f'Layer{layer_no}/'
                if module != []: Out_Path += f'{module}/'
                if phase != []: Out_Path += f'{phase}/'
                fname = fname

            if save_txt: filename = os.path.join(Out_Path, fname + '.txt')
            if save_hex: hexname = os.path.join(Out_Path, fname + '_hex.txt')

            Path(Out_Path).mkdir(parents=True, exist_ok=True)

            if torch.is_tensor(data):
                try:
                    data = data.detach()
                except:
                    pass
                data = data.numpy()

            if save_txt: outfile = open(filename, mode='w')
            if save_txt: outfile.write(f'{data.shape}\n')

            if save_hex: hexfile = open(hexname, mode='w')
            if save_hex: hexfile.write(f'{data.shape}\n')

            if len(data.shape) == 0:
                if save_txt: outfile.write(f'{data}\n')
                if save_hex: hexfile.write(f'{data}\n')
                pass
            elif len(data.shape) == 1:
                for x in data:
                    if save_txt: outfile.write(f'{x}\n')
                    if save_hex: hexfile.write(f'{convert_to_hex(x)}\n')
                    pass
            else:
                w, x, y, z = data.shape
                for _i in range(w):
                    for _j in range(x):
                        for _k in range(y):
                            for _l in range(z):
                                _value = data[_i, _j, _k, _l]
                                if save_txt: outfile.write(f'{_value}\n')
                                if save_hex: hexfile.write(f'{convert_to_hex(_value)}\n')
                                pass

            if save_hex: hexfile.close()
            if save_txt: outfile.close()

            if save_txt: print(f'\t\t--> Saved {filename}')
            if save_hex: print(f'\t\t--> Saved {hexname}')


def convert_to_hex(value):
    # We will Use Single-Precision, Truncated and Rounding into Brain Floating Point
    # IEEE754 Single-Precision: Sign=1, Exponent_Bit=8, Mantissa_Bit=23
    Exponent_Bit = 8
    Mantissa_Bit = 23
    Binary_Value1 = Floating2Binary(value, Exponent_Bit, Mantissa_Bit)
    Hexadecimal_Value1 = hex(int(Binary_Value1, 2))[2:]
    # Truncating and Rounding
    Floating_Hexadecimal = Truncating_Rounding(Hexadecimal_Value1)
    if len(Floating_Hexadecimal) < 4:
        Brain_Floating_Hexadecimal = Floating_Hexadecimal.zfill(4)
    else:
        Brain_Floating_Hexadecimal = Floating_Hexadecimal
    return Brain_Floating_Hexadecimal

'''
def Fmap_Reverse_Ordering(Out_CH, Out_Size, DataCH0_List, DataCH1_List):

    origin0 = pd.DataFrame(DataCH0_List)
    origin1 = pd.DataFrame(DataCH1_List)
    
    origin_ar0 = np.array(origin0)
    origin_ar1 = np.array(origin1)
    origin_size = np.size(origin_ar0)//16
    A = int(origin_size * 16)
    iter_13 = int(Out_Size / 13)
    
    concat_ar = np.repeat('0000',A*2).reshape(origin_size*2,16)

    convert_ar0 = origin_ar0.reshape(Out_CH//16, (Out_Size+iter_13*3)//4, Out_Size, 2, 4, 4)
    convert_ar1 = origin_ar1.reshape(Out_CH//16, (Out_Size+iter_13*3)//4, Out_Size, 2, 4, 4)

    concat_ar = np.concatenate( (convert_ar0[:,:,:,0], convert_ar0[:,:,:,1], convert_ar1[:,:,:,0], convert_ar1[:,:,:,1])  ,axis=4) #shape = (Out_CH//16, (Out_Size+iter_13*3)//4, Out_Size, 4, 16)
    concat_ar = concat_ar.transpose(0,4,2,1,3).reshape(Out_CH, Out_Size, (Out_Size+iter_13*3)//16, 16) #shape = (Out_CH//16, 16, Out_Size, (Out_Size+iter_13*3)//4, 4)
    final_ar = concat_ar[:,:,:,0:13]
    final_ar = final_ar.reshape(-1)
    
    def bfloat16_to_decimal(hex_str):
        # 32 비트 부동 소수점 값을 hex 문자열로 변환
        float32_hex = hex_str.ljust(8,'0')
        hex_data = bytes.fromhex(float32_hex)
        # hex 문자열을 부동 소수점 값으로 언팩
        decimal_value = struct.unpack('!f', hex_data)[0]

        return decimal_value

    # bfloat16_array = np.array(origin_ar, dtype=np.uint16)  # bfloat16 데이터
    decimal_array = np.vectorize(bfloat16_to_decimal)(final_ar)

    outlist = decimal_array.tolist()
        
    return outlist
'''
def bfloat16_to_decimal(hex_str):
    # 32 비트 부동 소수점 값을 hex 문자열로 변환
    float32_hex = hex_str.ljust(8,'0')
    hex_data = bytes.fromhex(float32_hex)
    # hex 문자열을 부동 소수점 값으로 언팩
    decimal_value = struct.unpack('!f', hex_data)[0]

    return decimal_value

def Fmap_Reverse_Ordering(Out_CH, Out_Size, DataCH0_List, DataCH1_List):

    origin0 = pd.DataFrame(DataCH0_List)
    origin1 = pd.DataFrame(DataCH1_List)
   
    origin_ar0 = np.array(origin0)
    origin_ar1 = np.array(origin1)
    origin_size = np.size(origin_ar0)//16
    A = int(origin_size * 16)
    iter_13 = int(Out_Size / 13)
   
    concat_ar = np.repeat('0000',A*2).reshape(origin_size*2,16)

    convert_ar0 = origin_ar0.reshape(Out_CH//16, (Out_Size+iter_13*3)//4, Out_Size, 2, 4, 4)
    convert_ar1 = origin_ar1.reshape(Out_CH//16, (Out_Size+iter_13*3)//4, Out_Size, 2, 4, 4)

    concat_ar = np.concatenate( (convert_ar0[:,:,:,0], convert_ar0[:,:,:,1], convert_ar1[:,:,:,0], convert_ar1[:,:,:,1])  ,axis=4) #shape = (Out_CH//16, (Out_Size+iter_13*3)//4, Out_Size, 4, 16)
    concat_ar = concat_ar.transpose(0,4,2,1,3).reshape(Out_CH, Out_Size, (Out_Size+iter_13*3)//16, 16) #shape = (Out_CH//16, 16, Out_Size, (Out_Size+iter_13*3)//4, 4)
    final_ar = concat_ar[:,:,:,0:13]
    final_ar = final_ar.reshape(-1)
   

    # after using numpy, reset the CPU affinity of the parent process so
    # that it will use all cores
    #os.system("taskset -p 0xff %d" % os.getpid())
   
    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # decimal_array = pool.map(bfloat16_to_decimal, final_ar)
    
    
   
    # bfloat16_array = np.array(origin_ar, dtype=np.uint16)  # bfloat16 데이터
   # decimal_array = np.vectorize(bfloat16_to_decimal)(final_ar)

   # outlist = decimal_array.tolist()
   
    def to_decimal(num):
        # Perform your desired conversion logic here if needed
        return float(num)  # This is a basic example assuming conversion to integer

    # decimal_array = np.vectorize(to_decimal(final_ar))
    decimal_vectorized = np.vectorize(to_decimal)

    # Apply the vectorized function to your array
    outlist = decimal_vectorized(final_ar)


    # outlist = decimal_array.tolist()
    outlist = outlist.tolist()
       
    return outlist
