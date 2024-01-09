import cv2
import math
import numpy as np
import pandas as pd
import os
import torch
import re
import struct
import time
from functools import lru_cache
import numba


# Combined Master-PhD in MSISLAB
def Image_Directory():
    Image = "../Main_Processing_Scratch/Pre_Image_Converted"
    os.makedirs(Image, exist_ok=True)


def Load_Image(Image_Path, frameWidth, frameHeight):
    img = cv2.imread(Image_Path)
    new_img = cv2.resize(img, (frameWidth, frameHeight))
    return new_img


def Write_Image(File_Path, Data_Wr):
    Image_Directory()
    print("\t --> " + " Image_Converted Directory is created!")
    with open(File_Path, mode="w") as file:
        for y in range(0, 416):
            for x in range(0, 416):
                s = Data_Wr[x, y]
                file.write(str(np.float32(s[0]) / 255.0) + "\n")
                file.write(str(np.float32(s[1]) / 255.0) + "\n")
                file.write(str(np.float32(s[2]) / 255.0) + "\n")


def Write_Image_into_BFP(File_Path, Data_Wr, Exponent_Bits, Mantissa_Bits):
    Image_Directory()
    with open(File_Path, mode="w") as file:
        for y in range(0, 416):
            for x in range(0, 416):
                s = Data_Wr[x, y]
                Channel1_Converted_Binary = Floating2Binary(np.int32(s[0]), Exponent_Bits, Mantissa_Bits)
                Channel2_Converted_Binary = Floating2Binary(np.int32(s[1]), Exponent_Bits, Mantissa_Bits)
                Channel3_Converted_Binary = Floating2Binary(np.int32(s[2]), Exponent_Bits, Mantissa_Bits)
                Channel1_Converted_Hex = hex(int(Channel1_Converted_Binary, 2))[2:]
                Channel2_Converted_Hex = hex(int(Channel2_Converted_Binary, 2))[2:]
                Channel3_Converted_Hex = hex(int(Channel3_Converted_Binary, 2))[2:]
                Channel1_Truncated_Rounded_Hex = Truncating_Rounding(Channel1_Converted_Hex)
                Channel2_Truncated_Rounded_Hex = Truncating_Rounding(Channel2_Converted_Hex)
                Channel3_Truncated_Rounded_Hex = Truncating_Rounding(Channel3_Converted_Hex)
                file.write(str(Channel1_Truncated_Rounded_Hex) + "\n")
                file.write(str(Channel2_Truncated_Rounded_Hex) + "\n")
                file.write(str(Channel3_Truncated_Rounded_Hex) + "\n")


def Write_Image_into_FP32(File_Path, Data_Wr, Exponent_Bits, Mantissa_Bits):
    Image_Directory()
    with open(File_Path, mode="w") as file:
        for y in range(0, 416):
            for x in range(0, 416):
                s = Data_Wr[x, y]
                Channel1_Converted_Binary = Floating2Binary(np.int32(s[0]), Exponent_Bits, Mantissa_Bits)
                Channel2_Converted_Binary = Floating2Binary(np.int32(s[1]), Exponent_Bits, Mantissa_Bits)
                Channel3_Converted_Binary = Floating2Binary(np.int32(s[2]), Exponent_Bits, Mantissa_Bits)
                Channel1_Converted_Hex = hex(int(Channel1_Converted_Binary, 2))[2:].upper()
                Channel2_Converted_Hex = hex(int(Channel2_Converted_Binary, 2))[2:].upper()
                Channel3_Converted_Hex = hex(int(Channel3_Converted_Binary, 2))[2:].upper()
                file.write(str(Channel1_Converted_Hex) + "\n")
                file.write(str(Channel2_Converted_Hex) + "\n")
                file.write(str(Channel3_Converted_Hex) + "\n")


def Read_Image(File_Path):
    Image_List = []
    with open(File_Path, "r") as file:
        for line in file:
            Image_List.append(line.strip())
    return Image_List


def Read_Weight(File_List, Read_Folder_Path):
    # Read all the Bias from a Bias_Folder
    Conv_Weight_List = {}

    List_Sorted = sorted(File_List)  # Sort the file names alphabetically

    for i, file in enumerate(List_Sorted):
        Read_File_Path = os.path.join(Read_Folder_Path, file)

        with open(Read_File_Path, mode="r") as file_r:
            Input = file_r.read()

        Input_List = [np.float32(Value) for Value in Input.split()]

        Conv_Weight_List[f"Weight_List{i}"] = Input_List

    # Separate List By List
    Weight_List0 = Conv_Weight_List["Weight_List0"]
    Weight_List1 = Conv_Weight_List["Weight_List1"]
    Weight_List2 = Conv_Weight_List["Weight_List2"]
    Weight_List3 = Conv_Weight_List["Weight_List3"]
    Weight_List4 = Conv_Weight_List["Weight_List4"]
    Weight_List5 = Conv_Weight_List["Weight_List5"]
    Weight_List6 = Conv_Weight_List["Weight_List6"]
    Weight_List7 = Conv_Weight_List["Weight_List7"]
    Weight_List8 = Conv_Weight_List["Weight_List8"]

    return Weight_List0, Weight_List1, Weight_List2, Weight_List3, \
        Weight_List4, Weight_List5, Weight_List6, Weight_List7, Weight_List8


def Read_Bias(File_List, Read_Folder_Path):
    # Read all the Bias from a Bias_Folder
    Bias = {}
    List_Sorted = sorted(File_List)  # Sort the file names alphabetically

    for i, file in enumerate(List_Sorted):
        Read_File_Path = os.path.join(Read_Folder_Path, file)

        with open(Read_File_Path, mode="r") as file_r:
            Input = file_r.read()

        Image_List = [np.float32(Value) for Value in Input.split()]
        Bias[f"Bias{i}"] = Image_List

    # Separate List By List
    Bias_8 = Bias["Bias0"]
    return Bias_8


def Read_BN_Parameters(File_List, Read_Folder_Path):
    # Read all the Bias from a Bias_Folder
    BN_Params_List = {}

    List_Sorted = sorted(File_List)  # Sort the file names alphabetically

    for i, file in enumerate(List_Sorted):
        Read_File_Path = os.path.join(Read_Folder_Path, file)

        with open(Read_File_Path, mode="r") as file_r:
            Input = file_r.read()

        BN_Params = [np.float32(Value) for Value in Input.split()]

        BN_Params_List[f"BN_Params_List{i}"] = BN_Params

    # Separate List By List
    BN_Params_List0 = BN_Params_List["BN_Params_List0"]
    BN_Params_List1 = BN_Params_List["BN_Params_List1"]
    BN_Params_List2 = BN_Params_List["BN_Params_List2"]
    BN_Params_List3 = BN_Params_List["BN_Params_List3"]
    BN_Params_List4 = BN_Params_List["BN_Params_List4"]
    BN_Params_List5 = BN_Params_List["BN_Params_List5"]
    BN_Params_List6 = BN_Params_List["BN_Params_List6"]
    BN_Params_List7 = BN_Params_List["BN_Params_List7"]

    return BN_Params_List0, BN_Params_List1, BN_Params_List2, BN_Params_List3, \
        BN_Params_List4, BN_Params_List5, BN_Params_List6, BN_Params_List7


def Read_Image_into_BFP(Data, Exponent_Bit, Mantissa_Bit):
    origin_ar = np.array(Data)
    def decimal_to_ieee754(decimal):
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

    trans_ar = np.vectorize(decimal_to_ieee754)(origin_ar)
    Image_List = trans_ar.tolist()
    return Image_List


def Read_Image_into_FP32(Data, Exponent_Bit, Mantissa_Bit):
    Image_List = []
    for Value in Data:
        Binary_Value = Floating2Binary(np.float32(Value), Exponent_Bit, Mantissa_Bit)
        Hexadecimal_Value = hex(int(Binary_Value, 2))[2:].upper()
        Image_List.append(str(Hexadecimal_Value).zfill(8))
    return Image_List


def Read_Weight_into_Bfloat16(Data_Tensor, Exponent_Bit, Mantissa_Bit): 
    List_Sorted = Data_Tensor.flatten().tolist()
    origin_ar = np.array(List_Sorted)
    def decimal_to_ieee754(decimal):
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
    trans_ar = np.vectorize(decimal_to_ieee754)(origin_ar)
    Hex_List = trans_ar.tolist()
    # Input_List = [np.float32(Value) for Value in List_Sorted]
    # Hex_List = []
    # for Value in Input_List:
    #     Binary_Value = Floating2Binary(Value, Exponent_Bit, Mantissa_Bit)
    #     Hexadecimal_Value = hex(int(Binary_Value, 2))[2:]
    #     Truncated_Rounded_Hex = Truncating_Rounding(Hexadecimal_Value)
    #     Hex_List.append(Truncated_Rounded_Hex)
    return Hex_List

def Read_Weight_into_FP32(Data_Tensor, Exponent_Bit, Mantissa_Bit):
    List_Sorted = Data_Tensor.flatten().tolist()
    Input_List = [np.float32(Value) for Value in List_Sorted]
    Hex_List = []
    for Value in Input_List:
        Binary_Value = Floating2Binary(Value, Exponent_Bit, Mantissa_Bit)
        Hexadecimal_Value = hex(int(Binary_Value, 2))[2:].upper()
        Hex_List.append(Hexadecimal_Value)

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

def Read_Bias_into_FP32(Data_List, Exponent_Bit, Mantissa_Bit):
    # Read all the Bias from a Bias_Folder

    List_Sorted = Data_List  # Sort the file names alphabetically

    Input_List = [np.float32(Value) for Value in List_Sorted]
    Hex_List = []
    for Value in Input_List:
        Binary_Value = Floating2Binary(Value, Exponent_Bit, Mantissa_Bit)
        Hexadecimal_Value = hex(int(Binary_Value, 2))[2:].upper()
        Hex_List.append(Hexadecimal_Value)

    return Hex_List


def Read_Bias_into_FP32(File_List, Read_Folder_Path, Exponent_Bit, Mantissa_Bit):
    # Read all the Bias from a Bias_Folder
    Bias_List = {}

    List_Sorted = sorted(File_List)  # Sort the file names alphabetically

    for i, file in enumerate(List_Sorted):
        Read_File_Path = os.path.join(Read_Folder_Path, file)

        with open(Read_File_Path, mode="r") as file_r:
            Input = file_r.read()

        Input_List = [np.float32(Value) for Value in Input.split()]
        Hex_List = []
        for Value in Input_List:
            Binary_Value = Floating2Binary(Value, Exponent_Bit, Mantissa_Bit)
            Hexadecimal_Value = hex(int(Binary_Value, 2))[2:].upper()
            Hex_List.append(Hexadecimal_Value)
        Bias_List[f"Bias_List{i}"] = Hex_List

    # Extract Weight_List1 from the dictionary
    Bias_List0 = Bias_List["Bias_List0"]

    return Bias_List0


def Read_BN_Parameters_into_BFP(File_List, Read_Folder_Path, Exponent_Bit, Mantissa_Bit):
    # Read all the Bias from a Bias_Folder
    BN_Params_List = {}

    List_Sorted = sorted(File_List)  # Sort the file names alphabetically

    for i, file in enumerate(List_Sorted):
        Read_File_Path = os.path.join(Read_Folder_Path, file)

        with open(Read_File_Path, mode="r") as file_r:
            Input = file_r.read()

        BN_Params = [np.float32(Value) for Value in Input.split()]
        Hex_List = []
        for Value in BN_Params:
            Binary_Value = Floating2Binary(Value, Exponent_Bit, Mantissa_Bit)
            Hexadecimal_Value = hex(int(Binary_Value, 2))[2:]
            Truncated_Rounded_Hex = Truncating_Rounding(Hexadecimal_Value)
            Hex_List.append(Truncated_Rounded_Hex)
        BN_Params_List[f"BN_Params_List{i}"] = Hex_List

    # Separate List By List
    BN_Params_List0 = BN_Params_List["BN_Params_List0"]
    BN_Params_List1 = BN_Params_List["BN_Params_List1"]
    BN_Params_List2 = BN_Params_List["BN_Params_List2"]
    BN_Params_List3 = BN_Params_List["BN_Params_List3"]
    BN_Params_List4 = BN_Params_List["BN_Params_List4"]
    BN_Params_List5 = BN_Params_List["BN_Params_List5"]
    BN_Params_List6 = BN_Params_List["BN_Params_List6"]
    BN_Params_List7 = BN_Params_List["BN_Params_List7"]

    return BN_Params_List0, BN_Params_List1, BN_Params_List2, BN_Params_List3, \
        BN_Params_List4, BN_Params_List5, BN_Params_List6, BN_Params_List7


def Read_BN_Parameters_into_FP32(File_List, Read_Folder_Path, Exponent_Bit, Mantissa_Bit):
    # Read all the Bias from a Bias_Folder
    BN_Params_List = {}

    List_Sorted = sorted(File_List)  # Sort the file names alphabetically

    for i, file in enumerate(List_Sorted):
        Read_File_Path = os.path.join(Read_Folder_Path, file)

        with open(Read_File_Path, mode="r") as file_r:
            Input = file_r.read()

        BN_Params = [np.float32(Value) for Value in Input.split()]
        Hex_List = []
        for Value in BN_Params:
            Binary_Value = Floating2Binary(Value, Exponent_Bit, Mantissa_Bit)
            Hexadecimal_Value = hex(int(Binary_Value, 2))[2:].upper()
            Hex_List.append(Hexadecimal_Value)
        BN_Params_List[f"BN_Params_List{i}"] = Hex_List

    # Separate List By List
    BN_Params_List0 = BN_Params_List["BN_Params_List0"]
    BN_Params_List1 = BN_Params_List["BN_Params_List1"]
    BN_Params_List2 = BN_Params_List["BN_Params_List2"]
    BN_Params_List3 = BN_Params_List["BN_Params_List3"]
    BN_Params_List4 = BN_Params_List["BN_Params_List4"]
    BN_Params_List5 = BN_Params_List["BN_Params_List5"]
    BN_Params_List6 = BN_Params_List["BN_Params_List6"]
    BN_Params_List7 = BN_Params_List["BN_Params_List7"]

    return BN_Params_List0, BN_Params_List1, BN_Params_List2, BN_Params_List3, \
        BN_Params_List4, BN_Params_List5, BN_Params_List6, BN_Params_List7

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


def Microcode(Read_Path):
    Microcode_List = []

    with open(Read_Path, 'r') as Micro_File:
        for Microcode in Micro_File:
            # Remove all the character from // and replace with ''
            Microcode = re.sub(r'//.*', '', Microcode)
            # Searching for '0' or '1'
            Microcode = re.findall(r'\b[01]+\b', Microcode)
            # All Binaries can store in a single list
            Microcode_List.extend(Microcode)
    return Microcode_List


# Hardware_ReOrdering: Forward
# Original from Sangbo Park

def Separated_Fmap_DDR_Channel(Data):
    origin = Data
    origin_ar = np.array(origin)
    origin_size = np.size(origin_ar)
    origin_ar = origin_ar.reshape(int(origin_size / 4), 4)

    zero = "0000000000000000000000000000000000000000000000000000000000000000"

    data_1 = np.repeat(zero, int(origin_size / 2)).reshape(int(origin_size / 4), 2)
    data_2 = np.repeat(zero, int(origin_size / 2)).reshape(int(origin_size / 4), 2)

    # for i in range(0, int(origin_size / 4)):
    #     data_1[i][0] = origin_ar[i][0]
    #     data_1[i][1] = origin_ar[i][1]
    #     data_2[i][0] = origin_ar[i][2]
    #     data_2[i][1] = origin_ar[i][3]

    data_1 = origin_ar[:,0:2]
    data_2 = origin_ar[:,2:4]


    # 16words
    df1 = pd.DataFrame(data_1.reshape(int(origin_size / 2)))
    df2 = pd.DataFrame(data_2.reshape(int(origin_size / 2)))

    return df1, df2

# Soft2Hardware Re-Ordering: Forward
# Output_Channel = 3 --> 16 = 3 + 13 (Filling 13 Channels More)
# def Fmap_Hardware_ReOrdering_Layer0(Out_Channel, Data_List):
#     Output_Channel = Out_Channel
#     origin = pd.DataFrame(Data_List)
#     origin_ar = np.array(origin)
#     origin_ar = origin_ar.reshape(3, 416, 416)

#     zero = '0000'
#     zero_ar = np.repeat(zero, 2249728)  # 13*416*416
#     zero_ar = zero_ar.reshape(13, 416, 416)

#     # makes input channel 16 (3+13)
#     origin_ar = np.concatenate((origin_ar, zero_ar), axis=0)

#     origin_size = np.size(origin_ar)
#     Fmap_size = int((origin_size / Output_Channel) ** (1 / 2))

#     zero = "0000"

#     # original fmap shape
#     origin_ar = origin_ar.reshape(Output_Channel, Fmap_size, Fmap_size)

#     # change row and col each other(전치)
#     for i in range(0, Output_Channel):
#         origin_ar[i] = origin_ar[i].T

#     origin_ar = origin_ar.reshape(Output_Channel, Fmap_size, Fmap_size)
#     iter_13 = int(Fmap_size / 13)

#     concat_ar = np.repeat(zero, Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3)).reshape(Output_Channel, (
#             Fmap_size + iter_13 * 3), Fmap_size)

#     for i in range(0, Output_Channel):
#         for j in range(0, iter_13):
#             for k in range(0, Fmap_size):
#                 concat_ar[i][j * 16 + 0][k] = origin_ar[i][j * 13 + 0][k]
#                 concat_ar[i][j * 16 + 1][k] = origin_ar[i][j * 13 + 1][k]
#                 concat_ar[i][j * 16 + 2][k] = origin_ar[i][j * 13 + 2][k]
#                 concat_ar[i][j * 16 + 3][k] = origin_ar[i][j * 13 + 3][k]
#                 concat_ar[i][j * 16 + 4][k] = origin_ar[i][j * 13 + 4][k]
#                 concat_ar[i][j * 16 + 5][k] = origin_ar[i][j * 13 + 5][k]
#                 concat_ar[i][j * 16 + 6][k] = origin_ar[i][j * 13 + 6][k]
#                 concat_ar[i][j * 16 + 7][k] = origin_ar[i][j * 13 + 7][k]
#                 concat_ar[i][j * 16 + 8][k] = origin_ar[i][j * 13 + 8][k]
#                 concat_ar[i][j * 16 + 9][k] = origin_ar[i][j * 13 + 9][k]
#                 concat_ar[i][j * 16 + 10][k] = origin_ar[i][j * 13 + 10][k]
#                 concat_ar[i][j * 16 + 11][k] = origin_ar[i][j * 13 + 11][k]
#                 concat_ar[i][j * 16 + 12][k] = origin_ar[i][j * 13 + 12][k]
#                 concat_ar[i][j * 16 + 13][k] = origin_ar[i][j * 13 + 12][k]
#                 concat_ar[i][j * 16 + 14][k] = origin_ar[i][j * 13 + 12][k]
#                 concat_ar[i][j * 16 + 15][k] = origin_ar[i][j * 13 + 12][k]
#     concat_ar = concat_ar.reshape(Output_Channel * (Fmap_size + iter_13 * 3), Fmap_size, 1)

#     four_ar1 = np.repeat(zero, int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 4)).reshape(
#         int(Output_Channel * (Fmap_size + iter_13 * 3) / 4), Fmap_size, 1)
#     four_ar2 = np.repeat(zero, int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 4)).reshape(
#         int(Output_Channel * (Fmap_size + iter_13 * 3) / 4), Fmap_size, 1)
#     four_ar3 = np.repeat(zero, int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 4)).reshape(
#         int(Output_Channel * (Fmap_size + iter_13 * 3) / 4), Fmap_size, 1)
#     four_ar4 = np.repeat(zero, int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 4)).reshape(
#         int(Output_Channel * (Fmap_size + iter_13 * 3) / 4), Fmap_size, 1)

#     for i in range(0, int(Output_Channel * (Fmap_size + iter_13 * 3) / 4)):
#         for j in range(0, Fmap_size):
#             four_ar1[i][j][0] = concat_ar[4 * i][j][0]
#             four_ar2[i][j][0] = concat_ar[4 * i + 1][j][0]
#             four_ar3[i][j][0] = concat_ar[4 * i + 2][j][0]
#             four_ar4[i][j][0] = concat_ar[4 * i + 3][j][0]

#     four_ar1 = four_ar1.reshape(Output_Channel, int(Fmap_size * (Fmap_size + iter_13 * 3) / 4), 1)
#     four_ar2 = four_ar2.reshape(Output_Channel, int(Fmap_size * (Fmap_size + iter_13 * 3) / 4), 1)
#     four_ar3 = four_ar3.reshape(Output_Channel, int(Fmap_size * (Fmap_size + iter_13 * 3) / 4), 1)
#     four_ar4 = four_ar4.reshape(Output_Channel, int(Fmap_size * (Fmap_size + iter_13 * 3) / 4), 1)

#     fmap1 = np.repeat(zero, int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 4)).reshape(
#         int(Output_Channel / 16), int(Fmap_size * (Fmap_size + iter_13 * 3) / 4), 16)
#     fmap2 = np.repeat(zero, int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 4)).reshape(
#         int(Output_Channel / 16), int(Fmap_size * (Fmap_size + iter_13 * 3) / 4), 16)
#     fmap3 = np.repeat(zero, int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 4)).reshape(
#         int(Output_Channel / 16), int(Fmap_size * (Fmap_size + iter_13 * 3) / 4), 16)
#     fmap4 = np.repeat(zero, int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 4)).reshape(
#         int(Output_Channel / 16), int(Fmap_size * (Fmap_size + iter_13 * 3) / 4), 16)

#     # concat 4 input channel
#     for i in range(0, int(Output_Channel / 16)):
#         fmap1[i] = np.concatenate((four_ar1[16 * i + 0], four_ar1[16 * i + 1], four_ar1[16 * i + 2],
#                                    four_ar1[16 * i + 3], four_ar2[16 * i + 0], four_ar2[16 * i + 1],
#                                    four_ar2[16 * i + 2], four_ar2[16 * i + 3], four_ar3[16 * i + 0],
#                                    four_ar3[16 * i + 1], four_ar3[16 * i + 2], four_ar3[16 * i + 3],
#                                    four_ar4[16 * i + 0], four_ar4[16 * i + 1], four_ar4[16 * i + 2],
#                                    four_ar4[16 * i + 3]), axis=1)
#         fmap2[i] = np.concatenate((four_ar1[16 * i + 4], four_ar1[16 * i + 5], four_ar1[16 * i + 6],
#                                    four_ar1[16 * i + 7], four_ar2[16 * i + 4], four_ar2[16 * i + 5],
#                                    four_ar2[16 * i + 6], four_ar2[16 * i + 7], four_ar3[16 * i + 4],
#                                    four_ar3[16 * i + 5], four_ar3[16 * i + 6], four_ar3[16 * i + 7],
#                                    four_ar4[16 * i + 4], four_ar4[16 * i + 5], four_ar4[16 * i + 6],
#                                    four_ar4[16 * i + 7]), axis=1)
#         fmap3[i] = np.concatenate((four_ar1[16 * i + 8], four_ar1[16 * i + 9], four_ar1[16 * i + 10],
#                                    four_ar1[16 * i + 11], four_ar2[16 * i + 8], four_ar2[16 * i + 9],
#                                    four_ar2[16 * i + 10], four_ar2[16 * i + 11], four_ar3[16 * i + 8],
#                                    four_ar3[16 * i + 9], four_ar3[16 * i + 10], four_ar3[16 * i + 11],
#                                    four_ar4[16 * i + 8], four_ar4[16 * i + 9], four_ar4[16 * i + 10],
#                                    four_ar4[16 * i + 11]), axis=1)
#         fmap4[i] = np.concatenate((four_ar1[16 * i + 12], four_ar1[16 * i + 13], four_ar1[16 * i + 14],
#                                    four_ar1[16 * i + 15], four_ar2[16 * i + 12], four_ar2[16 * i + 13],
#                                    four_ar2[16 * i + 14], four_ar2[16 * i + 15], four_ar3[16 * i + 12],
#                                    four_ar3[16 * i + 13], four_ar3[16 * i + 14], four_ar3[16 * i + 15],
#                                    four_ar4[16 * i + 12], four_ar4[16 * i + 13], four_ar4[16 * i + 14],
#                                    four_ar4[16 * i + 15]), axis=1)

#     # print(origin_size)
#     # print(fmap1.shape)

#     trans_ar = np.repeat(zero, Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3)).reshape(
#         int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 16), 16)
#     for i in range(0, int(Output_Channel / 16)):
#         for ch in range(0, 16):
#             for pix in range(0, int(Fmap_size * (Fmap_size + iter_13 * 3) / 4)):
#                 trans_ar[i * Fmap_size * (Fmap_size + iter_13 * 3) + pix * 4][ch] = fmap1[i][pix][ch]
#                 trans_ar[i * Fmap_size * (Fmap_size + iter_13 * 3) + pix * 4 + 1][ch] = fmap2[i][pix][ch]
#                 trans_ar[i * Fmap_size * (Fmap_size + iter_13 * 3) + pix * 4 + 2][ch] = fmap3[i][pix][ch]
#                 trans_ar[i * Fmap_size * (Fmap_size + iter_13 * 3) + pix * 4 + 3][ch] = fmap4[i][pix][ch]
               
#     Fmap_List = []
#     Fmap_List.clear() 
#     for value in trans_ar:
#         merge = ''.join(value)
#         Fmap_List.append(merge)
#     df = pd.DataFrame(Fmap_List)
#     # print(df)
#     df1, df2 = Separated_Fmap_DDR_Channel(df)
#     # print(type(df1))
#     # df1.to_csv(Write_Path_Ch0, index=False, header=False, sep='\t')
#     # df2.to_csv(Write_Path_Ch1, index=False, header=False, sep='\t')
#     Image_Channel0 = df1.values.tolist()
#     Image_Channel1 = df2.values.tolist()
#     Image_Layer0 = [Image_Channel0, Image_Channel1]
#     return Image_Layer0




def Fmap_Hardware_ReOrdering_Layer0(Output_Channel, Data_List):

    Data_List_ = Data_List + ["0000"] * 2249728

    origin = pd.DataFrame(Data_List_)
    origin_ar = np.array(origin)
    origin_size = np.size(origin_ar)
    Fmap_size = int((origin_size/Output_Channel)**(1/2))
    iter_13 = Fmap_size//13

    # original fmap shape
    origin_ar = origin_ar.reshape(Output_Channel,Fmap_size,Fmap_size).transpose(0,2,1)
    origin_ar = origin_ar.reshape(Output_Channel*iter_13,13,Fmap_size)
    origin_ar = np.concatenate( (origin_ar,origin_ar[:,12:13],origin_ar[:,12:13],origin_ar[:,12:13]), axis=1 ).reshape(Output_Channel//4,4, iter_13*4,4, Fmap_size) #(Output_Channel, iter_13*16, Fmap_size)
    origin_ar = origin_ar.transpose(0,2,4,3,1) #(Output_Channel//4, iter_13*4, Fmap_size, 4(iter_13*4), 4(Output_Channel//16))
    origin_ar = origin_ar.reshape(Output_Channel//16,4, iter_13*4*Fmap_size, 16)

    final_ar1 = np.concatenate((origin_ar[:,0],origin_ar[:,1]), axis=2).reshape(Output_Channel*iter_13*Fmap_size//2, 16)
    final_ar2 = np.concatenate((origin_ar[:,2],origin_ar[:,3]), axis=2).reshape(Output_Channel*iter_13*Fmap_size//2, 16)

    final_ar = np.concatenate((final_ar1,final_ar2), axis=0)

    final_list = []
    final_list.clear()
    for value in final_ar:
        Result = ''.join(value)
        final_list.append(Result)

    df = pd.DataFrame(final_list)
    final_ar = np.array(df).reshape(2,Output_Channel*iter_13*Fmap_size//2)
    df1 = pd.DataFrame(final_ar[0])
    df2 = pd.DataFrame(final_ar[1])

    Image_Channel0 = df1.values.tolist()
    Image_Channel1 = df2.values.tolist()
    Image_Layer0 = [Image_Channel0, Image_Channel1]

    return Image_Layer0




# Soft2Hardware Re-Ordering: Forward
# Output_Channel = 512, 128, 1024
# ReOrdering Fmap for all the layer except layer0
def Fmap_Hardware_ReOrdering(Out_Channel, Data_List):
    Output_Channel = Out_Channel
    origin = pd.DataFrame(Data_List)
    origin_ar = np.array(origin)
    origin_size = np.size(origin_ar)
    Fmap_size = int((origin_size / Output_Channel) ** (1 / 2))

    zero = "0000"

    # original fmap shape
    origin_ar = origin_ar.reshape(Output_Channel, Fmap_size, Fmap_size)

    # change row and col each other(전치)
    for i in range(0, Output_Channel):
        origin_ar[i] = origin_ar[i].T

    origin_ar = origin_ar.reshape(Output_Channel, Fmap_size, Fmap_size)
    iter_13 = int(Fmap_size / 13)

    concat_ar = np.repeat(zero, Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3)).reshape(Output_Channel,
                                                                                                (Fmap_size + iter_13 * 3),
                                                                                                Fmap_size)

    for i in range(0, Output_Channel):
        for j in range(0, iter_13):
            for k in range(0, Fmap_size):
                concat_ar[i][j * 16 + 0][k] = origin_ar[i][j * 13 + 0][k]
                concat_ar[i][j * 16 + 1][k] = origin_ar[i][j * 13 + 1][k]
                concat_ar[i][j * 16 + 2][k] = origin_ar[i][j * 13 + 2][k]
                concat_ar[i][j * 16 + 3][k] = origin_ar[i][j * 13 + 3][k]
                concat_ar[i][j * 16 + 4][k] = origin_ar[i][j * 13 + 4][k]
                concat_ar[i][j * 16 + 5][k] = origin_ar[i][j * 13 + 5][k]
                concat_ar[i][j * 16 + 6][k] = origin_ar[i][j * 13 + 6][k]
                concat_ar[i][j * 16 + 7][k] = origin_ar[i][j * 13 + 7][k]
                concat_ar[i][j * 16 + 8][k] = origin_ar[i][j * 13 + 8][k]
                concat_ar[i][j * 16 + 9][k] = origin_ar[i][j * 13 + 9][k]
                concat_ar[i][j * 16 + 10][k] = origin_ar[i][j * 13 + 10][k]
                concat_ar[i][j * 16 + 11][k] = origin_ar[i][j * 13 + 11][k]
                concat_ar[i][j * 16 + 12][k] = origin_ar[i][j * 13 + 12][k]
                concat_ar[i][j * 16 + 13][k] = origin_ar[i][j * 13 + 12][k]
                concat_ar[i][j * 16 + 14][k] = origin_ar[i][j * 13 + 12][k]
                concat_ar[i][j * 16 + 15][k] = origin_ar[i][j * 13 + 12][k]
    concat_ar = concat_ar.reshape(Output_Channel * (Fmap_size + iter_13 * 3), Fmap_size, 1)

    four_ar1 = np.repeat(zero, int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 4)).reshape(
        int(Output_Channel * (Fmap_size + iter_13 * 3) / 4), Fmap_size, 1)
    four_ar2 = np.repeat(zero, int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 4)).reshape(
        int(Output_Channel * (Fmap_size + iter_13 * 3) / 4), Fmap_size, 1)
    four_ar3 = np.repeat(zero, int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 4)).reshape(
        int(Output_Channel * (Fmap_size + iter_13 * 3) / 4), Fmap_size, 1)
    four_ar4 = np.repeat(zero, int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 4)).reshape(
        int(Output_Channel * (Fmap_size + iter_13 * 3) / 4), Fmap_size, 1)

    for i in range(0, int(Output_Channel * (Fmap_size + iter_13 * 3) / 4)):
        for j in range(0, Fmap_size):
            four_ar1[i][j][0] = concat_ar[4 * i][j][0]
            four_ar2[i][j][0] = concat_ar[4 * i + 1][j][0]
            four_ar3[i][j][0] = concat_ar[4 * i + 2][j][0]
            four_ar4[i][j][0] = concat_ar[4 * i + 3][j][0]

    four_ar1 = four_ar1.reshape(Output_Channel, int(Fmap_size * (Fmap_size + iter_13 * 3) / 4), 1)
    four_ar2 = four_ar2.reshape(Output_Channel, int(Fmap_size * (Fmap_size + iter_13 * 3) / 4), 1)
    four_ar3 = four_ar3.reshape(Output_Channel, int(Fmap_size * (Fmap_size + iter_13 * 3) / 4), 1)
    four_ar4 = four_ar4.reshape(Output_Channel, int(Fmap_size * (Fmap_size + iter_13 * 3) / 4), 1)

    fmap1 = np.repeat(zero, int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 4)).reshape(
        int(Output_Channel / 16), int(Fmap_size * (Fmap_size + iter_13 * 3) / 4), 16)
    fmap2 = np.repeat(zero, int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 4)).reshape(
        int(Output_Channel / 16), int(Fmap_size * (Fmap_size + iter_13 * 3) / 4), 16)
    fmap3 = np.repeat(zero, int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 4)).reshape(
        int(Output_Channel / 16), int(Fmap_size * (Fmap_size + iter_13 * 3) / 4), 16)
    fmap4 = np.repeat(zero, int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 4)).reshape(
        int(Output_Channel / 16), int(Fmap_size * (Fmap_size + iter_13 * 3) / 4), 16)

    # concat 4 input channel
    for i in range(0, int(Output_Channel / 16)):
        fmap1[i] = np.concatenate(
            (four_ar1[16 * i + 0], four_ar1[16 * i + 1], four_ar1[16 * i + 2], four_ar1[16 * i + 3],
             four_ar2[16 * i + 0], four_ar2[16 * i + 1], four_ar2[16 * i + 2], four_ar2[16 * i + 3],
             four_ar3[16 * i + 0], four_ar3[16 * i + 1], four_ar3[16 * i + 2], four_ar3[16 * i + 3],
             four_ar4[16 * i + 0], four_ar4[16 * i + 1], four_ar4[16 * i + 2], four_ar4[16 * i + 3]),
            axis=1)
        fmap2[i] = np.concatenate(
            (four_ar1[16 * i + 4], four_ar1[16 * i + 5], four_ar1[16 * i + 6], four_ar1[16 * i + 7],
             four_ar2[16 * i + 4], four_ar2[16 * i + 5], four_ar2[16 * i + 6], four_ar2[16 * i + 7],
             four_ar3[16 * i + 4], four_ar3[16 * i + 5], four_ar3[16 * i + 6], four_ar3[16 * i + 7],
             four_ar4[16 * i + 4], four_ar4[16 * i + 5], four_ar4[16 * i + 6], four_ar4[16 * i + 7]),
            axis=1)
        fmap3[i] = np.concatenate(
            (four_ar1[16 * i + 8], four_ar1[16 * i + 9], four_ar1[16 * i + 10], four_ar1[16 * i + 11],
             four_ar2[16 * i + 8], four_ar2[16 * i + 9], four_ar2[16 * i + 10], four_ar2[16 * i + 11],
             four_ar3[16 * i + 8], four_ar3[16 * i + 9], four_ar3[16 * i + 10], four_ar3[16 * i + 11],
             four_ar4[16 * i + 8], four_ar4[16 * i + 9], four_ar4[16 * i + 10],
             four_ar4[16 * i + 11]), axis=1)
        fmap4[i] = np.concatenate((four_ar1[16 * i + 12], four_ar1[16 * i + 13], four_ar1[16 * i + 14],
                                   four_ar1[16 * i + 15], four_ar2[16 * i + 12], four_ar2[16 * i + 13],
                                   four_ar2[16 * i + 14], four_ar2[16 * i + 15], four_ar3[16 * i + 12],
                                   four_ar3[16 * i + 13], four_ar3[16 * i + 14], four_ar3[16 * i + 15],
                                   four_ar4[16 * i + 12], four_ar4[16 * i + 13], four_ar4[16 * i + 14],
                                   four_ar4[16 * i + 15]), axis=1)

    print(origin_size)
    print(fmap1.shape)

    trans_ar = np.repeat(zero, Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3)).reshape(
        int(Output_Channel * Fmap_size * (Fmap_size + iter_13 * 3) / 16), 16)
    for i in range(0, int(Output_Channel / 16)):
        for ch in range(0, 16):
            for pix in range(0, int(Fmap_size * (Fmap_size + iter_13 * 3) / 4)):
                trans_ar[i * Fmap_size * (Fmap_size + iter_13 * 3) + pix * 4][ch] = fmap1[i][pix][ch]
                trans_ar[i * Fmap_size * (Fmap_size + iter_13 * 3) + pix * 4 + 1][ch] = fmap2[i][pix][ch]
                trans_ar[i * Fmap_size * (Fmap_size + iter_13 * 3) + pix * 4 + 2][ch] = fmap3[i][pix][ch]
                trans_ar[i * Fmap_size * (Fmap_size + iter_13 * 3) + pix * 4 + 3][ch] = fmap4[i][pix][ch]
    # Modified New List
    
    Fmap_List = []
    Fmap_List.clear()
    for value in trans_ar:
        merge = ''.join(value)
        Fmap_List.append(merge)
    df = pd.DataFrame(Fmap_List)
    df1, df2 = Separated_Fmap_DDR_Channel(df)
    Image_Channel0 = df1.values.tolist()
    Image_Channel1 = df2.values.tolist()
    Image_Layer1_8 = [Image_Channel0, Image_Channel1]
    return Image_Layer1_8


# Original from Sangbo Park
def Separated_Weight_DDR_Channel(Data):
    # origin = pd.read_table(Read_Path, header=None)
    origin = Data
    origin_ar = np.array(origin)
    origin_size = np.size(origin_ar)
    origin_ar = origin_ar.reshape(int(origin_size / 160), 160)

    zero = "0000000000000000000000000000000000000000000000000000000000000000"

    data_1 = np.repeat(zero, int(origin_size / 2)).reshape(int(origin_size / 160), 80)
    data_2 = np.repeat(zero, int(origin_size / 2)).reshape(int(origin_size / 160), 80)

    # for i in range(0, int(origin_size / 160)):
    #     for j in range(0, 40):
    #         data_1[i][j] = origin_ar[i][j]
    #         data_1[i][j + 40] = origin_ar[i][j + 40]
    #         data_2[i][j] = origin_ar[i][j + 80]
    #         data_2[i][j + 40] = origin_ar[i][j + 120]

    data_1 = origin_ar[:,0:80]
    data_2 = origin_ar[:,80:160]

    # 16words
    df1 = pd.DataFrame(data_1.reshape(int(origin_size / 2)))
    df2 = pd.DataFrame(data_2.reshape(int(origin_size / 2)))
    return df1, df2


# Filter = 1024
# In_Channel_Num = 512
# YOLOv5 Inference Ordering: Forward
def Weight_Hardware_ReOrdering_Layer0(Filter_Num, In_Channel_Num, Data_List):
    # ---------------------------- Filter_Num, In_Channel_Num should be changed -----------------------
    Filter_Num = Filter_Num
    In_Channel_Num = In_Channel_Num
    Branch_Num = 1

    A = int(Filter_Num / 8)
    B = int(In_Channel_Num / 4)
    C = int((Filter_Num * In_Channel_Num * 9) / 16)
    D = int(Filter_Num * In_Channel_Num * 9)

    E = int(Filter_Num * 4)
    F = int(Filter_Num / 4)
    G = int(A * B * 20 * 16)
    H = int(A * B * 20)

    # origin = pd.read_table(Read_Path, header=None)
    origin = pd.DataFrame(Data_List)

    dequant = '0000'

    quant0 = '0000'
    zpy0 = '0000'

    quant1 = '0000'
    zpy1 = '0000'
    quant2 = '0000'
    zpy2 = '0000'

    quant3 = '0000'
    zpy3 = '0000'

    # ------------------------------- For Layer 0 ------------------------------------
    origin_ar = np.array(origin)
    origin_ar = origin_ar.reshape(16, 3, 9)

    zero = '0000'
    zero_ar = np.repeat(zero, 1872)  # 16*13*9
    zero_ar = zero_ar.reshape(16, 13, 9)

    temp = np.repeat(zero, 2304)  # 16*16*9
    temp = temp.reshape(16, 16, 9)
    for i in range(0, 16):
        temp[i] = np.concatenate((origin_ar[i], zero_ar[i]), axis=0)

    origin_ar = temp
    # --------------------------------------------------------------------------------

    # ------------------------------- For Other Layers -------------------------------
    # origin_ar = np.array(origin)
    # --------------------------------------------------------------------------------

    origin_size = np.size(origin_ar)
    kenel_size = int(origin_size / (Filter_Num * In_Channel_Num))

    if (kenel_size == 1):
        origin_ar = origin_ar.reshape(A, 2, 4, B, 4, 1)
        zero = '0000'
        zero_ar = np.repeat(zero, 6)
        zero_ar = zero_ar.reshape(6)
        temp_ar = np.repeat(zero, D).reshape(A, 2, 4, B, 4, 9)
        for i in range(0, A):
            for j in range(0, 2):
                for k in range(0, 4):
                    for l in range(0, B):
                        for m in range(0, 4):
                            temp_ar[i][j][k][l][m] = np.concatenate(
                                (zero_ar, origin_ar[i][j][k][l][m], origin_ar[i][j][k][l][m], origin_ar[i][j][k][l][m]),
                                axis=0)
        # concat 4 in_channel
        filter_ar = np.repeat(zero, D).reshape(A, 2, 4, B, 9, 4)
        for i in range(0, A):
            for j in range(0, 2):
                for k in range(0, 4):
                    for l in range(0, B):
                        filter_ar[i][j][k][l] = temp_ar[i][j][k][l].T
        # print('aaaa')
    elif (kenel_size == 9):
        origin_ar = origin_ar.reshape(A, 2, 4, B, 4, 9)  # Using "2" make 8 outch
        zero = '0000'
        # concat 4 in channel
        filter_ar = np.repeat(zero, origin_size).reshape(A, 2, 4, B, 9, 4)
        for fn in range(0, A):
            for fc2 in range(0, 2):
                for fc1 in range(0, 4):
                    for incn in range(0, B):
                        filter_ar[fn][fc2][fc1][incn] = origin_ar[fn][fc2][fc1][incn].T
        # print('ssss')

    # to concat 4 filter
    filter_ar2 = np.repeat(zero, D).reshape(A, 2, B, 9, 16)
    for fn in range(0, A):
        for fin in range(0, 2):
            for cn in range(0, B):
                for d in range(0, 9):
                    filter_ar2[fn][fin][cn][d] = np.concatenate((filter_ar[fn][fin][0][cn][d],
                                                                 filter_ar[fn][fin][1][cn][d],
                                                                 filter_ar[fn][fin][2][cn][d],
                                                                 filter_ar[fn][fin][3][cn][d]), axis=0)

    filter_ar3 = np.repeat(zero, D).reshape(A, B, 2, 9, 16)  # to concat filter twice
    filter_ar3 = filter_ar2.transpose(0, 2, 1, 3, 4)

    Wconvert_ar = filter_ar3.reshape(C, 16)

    zero = '0000'
    bias_ar = np.repeat(zero, Filter_Num).reshape(Filter_Num, 1)  # outchannel num == filter num
    active_ar = np.repeat(zero, 1).reshape(1, 1)

    dequant_ar = np.repeat(zero, 1).reshape(1, 1)
    quant0_ar = np.repeat(zero, 1).reshape(1, 1)
    zpy0_ar = np.repeat(zero, 1).reshape(1, 1)
    quant1_ar = np.repeat(zero, 1).reshape(1, 1)
    zpy1_ar = np.repeat(zero, 1).reshape(1, 1)
    quant2_ar = np.repeat(zero, 1).reshape(1, 1)
    zpy2_ar = np.repeat(zero, 1).reshape(1, 1)
    quant3_ar = np.repeat(zero, 1).reshape(1, 1)
    zpy3_ar = np.repeat(zero, 1).reshape(1, 1)

    active_ar = np.array('4120', dtype=str).reshape(1, 1)

    dequant_ar = np.array(dequant)
    quant0_ar = np.array(quant0)
    zpy0_ar = np.array(zpy0)
    quant1_ar = np.array(quant1)
    zpy1_ar = np.array(zpy1)
    quant2_ar = np.array(quant2)
    zpy2_ar = np.array(zpy2)
    quant3_ar = np.array(quant3)
    zpy3_ar = np.array(zpy3)

    space = np.repeat(zero, 1).reshape(1, 1)
    vector_array = np.repeat(zero, E).reshape(A, 2, 16)  # outchannel num/8,2,16

    if (Branch_Num == 1):
        for i in range(0, A):  # filter_num / 8
            vector_array[i][0] = np.concatenate((bias_ar[i * 8], bias_ar[i * 8 + 1], bias_ar[i * 8 + 2],
                                                 bias_ar[i * 8 + 3], active_ar[0], active_ar[0], active_ar[0],
                                                 active_ar[0],
                                                 bias_ar[i * 8 + 4], bias_ar[i * 8 + 5], bias_ar[i * 8 + 6],
                                                 bias_ar[i * 8 + 7], active_ar[0], active_ar[0], active_ar[0],
                                                 active_ar[0]), axis=0)
            vector_array[i][1] = np.concatenate(
                (space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]
                 , space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]), axis=0)
    elif (Branch_Num == 2):
        for i in range(0, A):  # filter_num / 8
            vector_array[i][0] = np.concatenate((bias_ar[i * 8], bias_ar[i * 8 + 1], bias_ar[i * 8 + 2],
                                                 bias_ar[i * 8 + 3], active_ar[0], active_ar[0], active_ar[0],
                                                 active_ar[0],
                                                 bias_ar[i * 8 + 4], bias_ar[i * 8 + 5], bias_ar[i * 8 + 6],
                                                 bias_ar[i * 8 + 7], active_ar[0], active_ar[0], active_ar[0],
                                                 active_ar[0]), axis=0)
            vector_array[i][1] = np.concatenate(
                (space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]
                 , space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]), axis=0)
    elif (Branch_Num == 3):
        for i in range(0, A):  # filter_num / 8
            vector_array[i][0] = np.concatenate((bias_ar[i * 8], bias_ar[i * 8 + 1], bias_ar[i * 8 + 2],
                                                 bias_ar[i * 8 + 3], active_ar[0], active_ar[0], active_ar[0],
                                                 active_ar[0],
                                                 bias_ar[i * 8 + 4], bias_ar[i * 8 + 5], bias_ar[i * 8 + 6],
                                                 bias_ar[i * 8 + 7], active_ar[0], active_ar[0], active_ar[0],
                                                 active_ar[0]), axis=0)
            vector_array[i][1] = np.concatenate(
                (space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]
                 , space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]), axis=0)
    elif (Branch_Num == 4):
        for i in range(0, A):  # filter_num / 8
            vector_array[i][0] = np.concatenate((bias_ar[i * 8], bias_ar[i * 8 + 1], bias_ar[i * 8 + 2],
                                                 bias_ar[i * 8 + 3], active_ar[0], active_ar[0], active_ar[0],
                                                 active_ar[0],
                                                 bias_ar[i * 8 + 4], bias_ar[i * 8 + 5], bias_ar[i * 8 + 6],
                                                 bias_ar[i * 8 + 7], active_ar[0], active_ar[0], active_ar[0],
                                                 active_ar[0]), axis=0)
            vector_array[i][1] = np.concatenate(
                (space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]
                 , space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]), axis=0)

    vector_array = vector_array.reshape(F, 16)
    zero = '0000'

    Wconvert_array = Wconvert_ar.reshape(A, B, 18, 16)  # fil/8, in/4
    biacsc_ar = np.repeat(zero, E).reshape(A, 1, 2, 16)
    biacsc_ar = np.array(vector_array).reshape(A, 1, 2, 16)
    concat_ar = np.repeat(zero, G).reshape(int(A / 2), B * 2, 20, 16)

    # print(Wconvert_array[0][0])

    for k in range(0, int(A / 2)):
        for i in range(0, B):
            concat_ar[k][2 * i] = np.concatenate((Wconvert_array[2 * k][i], biacsc_ar[2 * k][0]), axis=0)
            concat_ar[k][2 * i + 1] = np.concatenate((Wconvert_array[2 * k + 1][i], biacsc_ar[2 * k + 1][0]), axis=0)

    convert_ar = np.repeat(zero, G).reshape(H, 16)
    convert_ar = concat_ar.reshape(H, 16)

    
    Weight_List = []
    Weight_List.clear()
    for value in convert_ar:
        Result = ''.join(value)
        Weight_List.append(Result)

    df = pd.DataFrame(Weight_List)
    # df.to_csv(Write_Path, index=False, header=False, sep='\t')
    df1, df2 = Separated_Weight_DDR_Channel(df)
    Weight_Layer0_Channel0 = df1.values.tolist()
    Weight_Layer0_Channel1 = df2.values.tolist()
    Weight_Layer0 = [Weight_Layer0_Channel0, Weight_Layer0_Channel1]
    return Weight_Layer0

# YOLOv5 Inference Ordering: Forward
def Weight_Hardware_ReOrdering_OtherLayer(Filter_Num, In_Channel_Num, Data_List):
    # ---------------------------- Filter_Num, In_Channel_Num should be changed -----------------------
    Filter_Num = Filter_Num
    In_Channel_Num = In_Channel_Num
    Branch_Num = 1

    A = int(Filter_Num / 8)
    B = int(In_Channel_Num / 4)
    C = int((Filter_Num * In_Channel_Num * 9) / 16)
    D = int(Filter_Num * In_Channel_Num * 9)

    E = int(Filter_Num * 4)
    F = int(Filter_Num / 4)
    G = int(A * B * 20 * 16)
    H = int(A * B * 20)

    # ------------------------------ All files require a garbage value in the first line. (ex) Add "test" to the
    # first line---------------------------

    # origin = pd.read_table(Read_Path, header=None)
    origin = pd.DataFrame(Data_List)

    dequant = '0000'

    quant0 = '0000'
    zpy0 = '0000'

    quant1 = '0000'
    zpy1 = '0000'
    quant2 = '0000'
    zpy2 = '0000'

    quant3 = '0000'
    zpy3 = '0000'

    # ------------------------------- For Other Layers -------------------------------
    origin_ar = np.array(origin)
    # --------------------------------------------------------------------------------

    origin_size = np.size(origin_ar)
    kenel_size = int(origin_size / (Filter_Num * In_Channel_Num))

    if (kenel_size == 1):
        origin_ar = origin_ar.reshape(A, 2, 4, B, 4,
                                      1)  # (filter/8 ,2(아래로 반복), 4(4개있음), input channel/4, 4(4개를 가져감), kernel)
        zero = '0000'
        zero_ar = np.repeat(zero, 6)
        zero_ar = zero_ar.reshape(6)
        temp_ar = np.repeat(zero, D).reshape(A, 2, 4, B, 4, 9)
        for i in range(0, A):
            for j in range(0, 2):
                for k in range(0, 4):
                    for l in range(0, B):
                        for m in range(0, 4):
                            temp_ar[i][j][k][l][m] = np.concatenate(
                                (zero_ar, origin_ar[i][j][k][l][m], origin_ar[i][j][k][l][m], origin_ar[i][j][k][l][m]),
                                axis=0)
        # concat 4 in_channel
        filter_ar = np.repeat(zero, D).reshape(A, 2, 4, B, 9, 4)
        for i in range(0, A):
            for j in range(0, 2):
                for k in range(0, 4):
                    for l in range(0, B):
                        filter_ar[i][j][k][l] = temp_ar[i][j][k][l].T
        # print('aaaa')
    elif (kenel_size == 9):
        origin_ar = origin_ar.reshape(A, 2, 4, B, 4, 9)  # Using "2" make 8 outch
        zero = '0000'
        # concat 4 in channel
        filter_ar = np.repeat(zero, origin_size).reshape(A, 2, 4, B, 9, 4)
        for fn in range(0, A):
            for fc2 in range(0, 2):
                for fc1 in range(0, 4):
                    for incn in range(0, B):
                        filter_ar[fn][fc2][fc1][incn] = origin_ar[fn][fc2][fc1][incn].T
        # print('ssss')

    # to concat 4 filter
    filter_ar2 = np.repeat(zero, D).reshape(A, 2, B, 9, 16)
    for fn in range(0, A):
        for fin in range(0, 2):
            for cn in range(0, B):
                for d in range(0, 9):
                    filter_ar2[fn][fin][cn][d] = np.concatenate((filter_ar[fn][fin][0][cn][d],
                                                                 filter_ar[fn][fin][1][cn][d],
                                                                 filter_ar[fn][fin][2][cn][d],
                                                                 filter_ar[fn][fin][3][cn][d]), axis=0)

    filter_ar3 = np.repeat(zero, D).reshape(A, B, 2, 9, 16)  # to concat filter twice
    filter_ar3 = filter_ar2.transpose(0, 2, 1, 3, 4)

    Wconvert_ar = filter_ar3.reshape(C, 16)

    # ---------------------------------------------------------------------------------------------------------------

    zero = '0000'
    bias_ar = np.repeat(zero, Filter_Num).reshape(Filter_Num, 1)  # outchannel num == filter num
    active_ar = np.repeat(zero, 1).reshape(1, 1)

    dequant_ar = np.repeat(zero, 1).reshape(1, 1)
    quant0_ar = np.repeat(zero, 1).reshape(1, 1)
    zpy0_ar = np.repeat(zero, 1).reshape(1, 1)
    quant1_ar = np.repeat(zero, 1).reshape(1, 1)
    zpy1_ar = np.repeat(zero, 1).reshape(1, 1)
    quant2_ar = np.repeat(zero, 1).reshape(1, 1)
    zpy2_ar = np.repeat(zero, 1).reshape(1, 1)
    quant3_ar = np.repeat(zero, 1).reshape(1, 1)
    zpy3_ar = np.repeat(zero, 1).reshape(1, 1)

    # bias_ar = np.concatenate( ( np.array(bias), np.repeat(zero,3).reshape(3,1) ), axis=0 )
    # bias_ar = np.array(bias)
    active_ar = np.array('4120', dtype=str).reshape(1, 1)

    dequant_ar = np.array(dequant)
    quant0_ar = np.array(quant0)
    zpy0_ar = np.array(zpy0)
    quant1_ar = np.array(quant1)
    zpy1_ar = np.array(zpy1)
    quant2_ar = np.array(quant2)
    zpy2_ar = np.array(zpy2)
    quant3_ar = np.array(quant3)
    zpy3_ar = np.array(zpy3)

    space = np.repeat(zero, 1).reshape(1, 1)
    vector_array = np.repeat(zero, E).reshape(A, 2, 16)  # outchannel num/8,2,16

    if (Branch_Num == 1):
        for i in range(0, A):  # filter_num / 8
            vector_array[i][0] = np.concatenate((bias_ar[i * 8], bias_ar[i * 8 + 1], bias_ar[i * 8 + 2],
                                                 bias_ar[i * 8 + 3], active_ar[0], active_ar[0], active_ar[0],
                                                 active_ar[0],
                                                 bias_ar[i * 8 + 4], bias_ar[i * 8 + 5], bias_ar[i * 8 + 6],
                                                 bias_ar[i * 8 + 7], active_ar[0], active_ar[0], active_ar[0],
                                                 active_ar[0]), axis=0)
            vector_array[i][1] = np.concatenate(
                (space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]
                 , space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]), axis=0)
    elif (Branch_Num == 2):
        for i in range(0, A):  # filter_num / 8
            vector_array[i][0] = np.concatenate((bias_ar[i * 8], bias_ar[i * 8 + 1], bias_ar[i * 8 + 2],
                                                 bias_ar[i * 8 + 3], active_ar[0], active_ar[0], active_ar[0],
                                                 active_ar[0],
                                                 bias_ar[i * 8 + 4], bias_ar[i * 8 + 5], bias_ar[i * 8 + 6],
                                                 bias_ar[i * 8 + 7], active_ar[0], active_ar[0], active_ar[0],
                                                 active_ar[0]), axis=0)
            vector_array[i][1] = np.concatenate(
                (space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]
                 , space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]), axis=0)
    elif (Branch_Num == 3):
        for i in range(0, A):  # filter_num / 8
            vector_array[i][0] = np.concatenate((bias_ar[i * 8], bias_ar[i * 8 + 1], bias_ar[i * 8 + 2],
                                                 bias_ar[i * 8 + 3], active_ar[0], active_ar[0], active_ar[0],
                                                 active_ar[0],
                                                 bias_ar[i * 8 + 4], bias_ar[i * 8 + 5], bias_ar[i * 8 + 6],
                                                 bias_ar[i * 8 + 7], active_ar[0], active_ar[0], active_ar[0],
                                                 active_ar[0]), axis=0)
            vector_array[i][1] = np.concatenate(
                (space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]
                 , space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]), axis=0)
    elif (Branch_Num == 4):
        for i in range(0, A):  # filter_num / 8
            vector_array[i][0] = np.concatenate((bias_ar[i * 8], bias_ar[i * 8 + 1], bias_ar[i * 8 + 2],
                                                 bias_ar[i * 8 + 3], active_ar[0], active_ar[0], active_ar[0],
                                                 active_ar[0],
                                                 bias_ar[i * 8 + 4], bias_ar[i * 8 + 5], bias_ar[i * 8 + 6],
                                                 bias_ar[i * 8 + 7], active_ar[0], active_ar[0], active_ar[0],
                                                 active_ar[0]), axis=0)
            vector_array[i][1] = np.concatenate(
                (space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]
                 , space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]), axis=0)

    vector_array = vector_array.reshape(F, 16)
    zero = '0000'

    Wconvert_array = Wconvert_ar.reshape(A, B, 18, 16)  # fil/8, in/4
    biacsc_ar = np.repeat(zero, E).reshape(A, 1, 2, 16)
    biacsc_ar = np.array(vector_array).reshape(A, 1, 2, 16)
    concat_ar = np.repeat(zero, G).reshape(int(A / 2), B * 2, 20, 16)

    # print(Wconvert_array[0][0])
    # print(biacsc_ar)

    for k in range(0, int(A / 2)):
        for i in range(0, B):
            concat_ar[k][2 * i] = np.concatenate((Wconvert_array[2 * k][i], biacsc_ar[2 * k][0]), axis=0)
            concat_ar[k][2 * i + 1] = np.concatenate((Wconvert_array[2 * k + 1][i], biacsc_ar[2 * k + 1][0]), axis=0)

    convert_ar = np.repeat(zero, G).reshape(H, 16)
    convert_ar = concat_ar.reshape(H, 16)

    
    Weight_List = []
    Weight_List.clear()
    for value in convert_ar:
        Result = ''.join(value)
        Weight_List.append(Result)

    df = pd.DataFrame(Weight_List)
    # df.to_csv(Write_Path, index=False, header=False, sep='\t')
    df1, df2 = Separated_Weight_DDR_Channel(df)
    # df1.to_csv(Write_Path_Ch1, index=False, header=False, sep='\t')
    # df2.to_csv(Write_Path_Ch2, index=False, header=False, sep='\t')
    Weight_Layer1_7_Channel0 = df1.values.tolist()
    Weight_Layer1_7_Channel1 = df2.values.tolist()
    Weight_Layer1_7 = [Weight_Layer1_7_Channel0, Weight_Layer1_7_Channel1]
    return Weight_Layer1_7

# YOLOv5 Inference Ordering: Forward
def Weight_Hardware_ReOrdering_Layer8(Filter_Num, In_Channel_Num, Data_List, Bias_List):
    Filter_Num = Filter_Num
    In_Channel_Num = In_Channel_Num
    Branch_Num = 1

    A = int(Filter_Num / 8)
    B = int(In_Channel_Num / 4)
    C = int((Filter_Num * In_Channel_Num * 9) / 16)
    D = int(Filter_Num * In_Channel_Num * 9)

    E = int(Filter_Num * 4)
    F = int(Filter_Num / 4)
    G = int(A * B * 20 * 16)
    H = int(A * B * 20)

    # Padding Weight for 1024 * 3

    # origin = pd.read_table(Read_Path, header=None)
    # bias = pd.read_table(Read_Path, header=None)

    # Padding the Weight List
    Weight_Length = 131072
    padding_size = 3072
    current_length = len(Data_List)
    padding_needed = max(0, Weight_Length - current_length)
    padding_count = min(padding_needed, padding_size)
    Data_List += ["0000"] * padding_count

    # Padding the Bias List
    Bias_Length = 128
    padding_size = 3
    current_length = len(Bias_List)
    padding_needed = max(0, Bias_Length - current_length)
    padding_count = min(padding_needed, padding_size)
    Bias_List += ["0000"] * padding_count

    origin = pd.DataFrame(Data_List)
    bias = pd.DataFrame(Bias_List)

    dequant = '0000'

    quant0 = '0000'
    zpy0 = '0000'

    quant1 = '0000'
    zpy1 = '0000'
    quant2 = '0000'
    zpy2 = '0000'

    quant3 = '0000'
    zpy3 = '0000'

    # ------------------------------- For Other Layers -------------------------------
    origin_ar = np.array(origin)
    # --------------------------------------------------------------------------------

    origin_size = np.size(origin_ar)
    kenel_size = int(origin_size / (Filter_Num * In_Channel_Num))

    if (kenel_size == 1):
        origin_ar = origin_ar.reshape(A, 2, 4, B, 4, 1)
        zero = '0000'
        zero_ar = np.repeat(zero, 6)
        zero_ar = zero_ar.reshape(6)
        temp_ar = np.repeat(zero, D).reshape(A, 2, 4, B, 4, 9)
        for i in range(0, A):
            for j in range(0, 2):
                for k in range(0, 4):
                    for l in range(0, B):
                        for m in range(0, 4):
                            temp_ar[i][j][k][l][m] = np.concatenate(
                                (zero_ar, origin_ar[i][j][k][l][m], origin_ar[i][j][k][l][m], origin_ar[i][j][k][l][m]),
                                axis=0)
        # concat 4 in_channel
        filter_ar = np.repeat(zero, D).reshape(A, 2, 4, B, 9, 4)
        for i in range(0, A):
            for j in range(0, 2):
                for k in range(0, 4):
                    for l in range(0, B):
                        filter_ar[i][j][k][l] = temp_ar[i][j][k][l].T
        # print('aaaa')
    elif (kenel_size == 9):
        origin_ar = origin_ar.reshape(A, 2, 4, B, 4, 9)
        zero = '0000'
        # concat 4 in channel
        filter_ar = np.repeat(zero, origin_size).reshape(A, 2, 4, B, 9, 4)
        for fn in range(0, A):
            for fc2 in range(0, 2):
                for fc1 in range(0, 4):
                    for incn in range(0, B):
                        filter_ar[fn][fc2][fc1][incn] = origin_ar[fn][fc2][fc1][incn].T
        # print('ssss')

    # to concat 4 filter
    filter_ar2 = np.repeat(zero, D).reshape(A, 2, B, 9, 16)
    for fn in range(0, A):
        for fin in range(0, 2):
            for cn in range(0, B):
                for d in range(0, 9):
                    filter_ar2[fn][fin][cn][d] = np.concatenate(
                        (filter_ar[fn][fin][0][cn][d], filter_ar[fn][fin][1][cn][d],
                         filter_ar[fn][fin][2][cn][d],
                         filter_ar[fn][fin][3][cn][d]), axis=0)

    filter_ar3 = np.repeat(zero, D).reshape(A, B, 2, 9, 16)
    filter_ar3 = filter_ar2.transpose(0, 2, 1, 3, 4)

    Wconvert_ar = filter_ar3.reshape(C, 16)

    zero = '0000'
    bias_ar = np.repeat(zero, Filter_Num).reshape(Filter_Num, 1)
    active_ar = np.repeat(zero, 1).reshape(1, 1)

    dequant_ar = np.repeat(zero, 1).reshape(1, 1)
    quant0_ar = np.repeat(zero, 1).reshape(1, 1)
    zpy0_ar = np.repeat(zero, 1).reshape(1, 1)
    quant1_ar = np.repeat(zero, 1).reshape(1, 1)
    zpy1_ar = np.repeat(zero, 1).reshape(1, 1)
    quant2_ar = np.repeat(zero, 1).reshape(1, 1)
    zpy2_ar = np.repeat(zero, 1).reshape(1, 1)
    quant3_ar = np.repeat(zero, 1).reshape(1, 1)
    zpy3_ar = np.repeat(zero, 1).reshape(1, 1)

    # bias_ar = np.concatenate( ( np.array(bias), np.repeat(zero,3).reshape(3,1) ), axis=0 )
    bias_ar = np.array(bias)
    active_ar = np.array('4120', dtype=str).reshape(1, 1)

    dequant_ar = np.array(dequant)
    quant0_ar = np.array(quant0)
    zpy0_ar = np.array(zpy0)
    quant1_ar = np.array(quant1)
    zpy1_ar = np.array(zpy1)
    quant2_ar = np.array(quant2)
    zpy2_ar = np.array(zpy2)
    quant3_ar = np.array(quant3)
    zpy3_ar = np.array(zpy3)

    space = np.repeat(zero, 1).reshape(1, 1)
    vector_array = np.repeat(zero, E).reshape(A, 2, 16)  # outchannel num/8,2,16

    if (Branch_Num == 1):
        for i in range(0, A):  # filter_num / 8
            vector_array[i][0] = np.concatenate(
                (bias_ar[i * 8], bias_ar[i * 8 + 1], bias_ar[i * 8 + 2], bias_ar[i * 8 + 3],
                 active_ar[0], active_ar[0], active_ar[0], active_ar[0],
                 bias_ar[i * 8 + 4], bias_ar[i * 8 + 5], bias_ar[i * 8 + 6],
                 bias_ar[i * 8 + 7], active_ar[0], active_ar[0], active_ar[0],
                 active_ar[0]), axis=0)
            vector_array[i][1] = np.concatenate(
                (space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]
                 , space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]), axis=0)
    elif (Branch_Num == 2):
        for i in range(0, A):  # filter_num / 8
            vector_array[i][0] = np.concatenate(
                (bias_ar[i * 8], bias_ar[i * 8 + 1], bias_ar[i * 8 + 2], bias_ar[i * 8 + 3],
                 active_ar[0], active_ar[0], active_ar[0], active_ar[0],
                 bias_ar[i * 8 + 4], bias_ar[i * 8 + 5], bias_ar[i * 8 + 6],
                 bias_ar[i * 8 + 7], active_ar[0], active_ar[0], active_ar[0],
                 active_ar[0]), axis=0)
            vector_array[i][1] = np.concatenate(
                (space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]
                 , space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]), axis=0)
    elif (Branch_Num == 3):
        for i in range(0, A):  # filter_num / 8
            vector_array[i][0] = np.concatenate(
                (bias_ar[i * 8], bias_ar[i * 8 + 1], bias_ar[i * 8 + 2], bias_ar[i * 8 + 3],
                 active_ar[0], active_ar[0], active_ar[0], active_ar[0],
                 bias_ar[i * 8 + 4], bias_ar[i * 8 + 5], bias_ar[i * 8 + 6],
                 bias_ar[i * 8 + 7], active_ar[0], active_ar[0], active_ar[0],
                 active_ar[0]), axis=0)
            vector_array[i][1] = np.concatenate(
                (space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]
                 , space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]), axis=0)
    elif (Branch_Num == 4):
        for i in range(0, A):  # filter_num / 8
            vector_array[i][0] = np.concatenate(
                (bias_ar[i * 8], bias_ar[i * 8 + 1], bias_ar[i * 8 + 2], bias_ar[i * 8 + 3],
                 active_ar[0], active_ar[0], active_ar[0], active_ar[0],
                 bias_ar[i * 8 + 4], bias_ar[i * 8 + 5], bias_ar[i * 8 + 6],
                 bias_ar[i * 8 + 7], active_ar[0], active_ar[0], active_ar[0],
                 active_ar[0]), axis=0)
            vector_array[i][1] = np.concatenate(
                (space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]
                 , space[0], space[0], space[0], space[0], space[0], space[0], space[0], space[0]), axis=0)

    vector_array = vector_array.reshape(F, 16)
    zero = '0000'

    Wconvert_array = Wconvert_ar.reshape(A, B, 18, 16)  # fil/8, in/4
    biacsc_ar = np.repeat(zero, E).reshape(A, 1, 2, 16)
    biacsc_ar = np.array(vector_array).reshape(A, 1, 2, 16)
    concat_ar = np.repeat(zero, G).reshape(int(A / 2), B * 2, 20, 16)

    # print(Wconvert_array[0][0])
    # print(biacsc_ar)

    for k in range(0, int(A / 2)):
        for i in range(0, B):
            concat_ar[k][2 * i] = np.concatenate((Wconvert_array[2 * k][i], biacsc_ar[2 * k][0]), axis=0)
            concat_ar[k][2 * i + 1] = np.concatenate((Wconvert_array[2 * k + 1][i], biacsc_ar[2 * k + 1][0]), axis=0)

    convert_ar = np.repeat(zero, G).reshape(H, 16)
    convert_ar = concat_ar.reshape(H, 16)

    
    Weight_List = []
    Weight_List.clear()
    for value in convert_ar:
        Result = ''.join(value)
        Weight_List.append(Result)
    df = pd.DataFrame(Weight_List)
    df1, df2 = Separated_Weight_DDR_Channel(df)
    Weight_Layer8_Channel0 = df1.values.tolist()
    Weight_Layer8_Channel1 = df2.values.tolist()
    Weight_Layer8 = [Weight_Layer8_Channel0, Weight_Layer8_Channel1]
    return Weight_Layer8

'''
def New_Weight_Hardware_ReOrdering_Layer0(Filter_Num, In_Channel_Num, Data_List, Sub_List, Mul_List, Add_List, Iteration):
    # ---------------------------- Filter_Num, In_Channel_Num should be changed -----------------------
    # Filter_Num = 128
    # In_Channel_Num = 1024
    Filter_Num = Filter_Num
    In_Channel_Num = In_Channel_Num

    A = int(Filter_Num / 8)
    B = int(In_Channel_Num / 4)
    C = int((Filter_Num * In_Channel_Num * 9) / 16)
    D = int(Filter_Num * In_Channel_Num * 9)

    E = int(Filter_Num * 4)
    F = int(Filter_Num / 4)
    G = int(A * B * 20 * 16)
    H = int(A * B * 20)
    zero = '0000'

    origin = pd.DataFrame(Data_List)
    # -------------------------------------------------------------------------------
    # --------------------------------Activation and BatchNorm-----------------------
    batch_sub = pd.DataFrame(Sub_List)
    batch_mul = pd.DataFrame(Mul_List)
    batch_add = pd.DataFrame(Add_List)
    if Iteration == "1":
        bias_active_ar = np.repeat('0000', Filter_Num).reshape(Filter_Num, 1)  # outchannel num == filter num
    else:
        bias_active_ar = np.repeat('4120', Filter_Num).reshape(Filter_Num, 1)  # outchannel num == filter num 
    batch_sub_ar = np.repeat(zero, Filter_Num).reshape(Filter_Num, 1)
    batch_mul_ar = np.repeat(zero, Filter_Num).reshape(Filter_Num, 1)
    batch_add_ar = np.repeat(zero, Filter_Num).reshape(Filter_Num, 1)

    batch_sub_ar = np.array(batch_sub)
    batch_mul_ar = np.array(batch_mul)
    batch_add_ar = np.array(batch_add)

    # -------------------------------------------------------------------------------
    # ------------------------------- For Layer 0 ------------------------------------
    origin_ar = np.array(origin)
    origin_ar = origin_ar.reshape(16, 3, 9)

    zero = '0000'
    zero_ar = np.repeat(zero, 1872)  # 16*13*9
    zero_ar = zero_ar.reshape(16, 13, 9)

    temp = np.repeat(zero, 2304)  # 16*16*9
    temp = temp.reshape(16, 16, 9)
    for i in range(0, 16):
        temp[i] = np.concatenate((origin_ar[i], zero_ar[i]), axis=0)

    origin_ar = temp
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    origin_size = np.size(origin_ar)
    # print(origin_size)
    kenel_size = int(origin_size / (Filter_Num * In_Channel_Num))
    # print(kenel_size)

    if kenel_size == 1:
        origin_ar = origin_ar.reshape(A, 2, 4, B, 4, 1)
        zero = '0000'
        zero_ar = np.repeat(zero, 6)
        zero_ar = zero_ar.reshape(6)
        temp_ar = np.repeat(zero, D).reshape(A, 2, 4, B, 4, 9)
        for i in range(0, A):
            for j in range(0, 2):
                for k in range(0, 4):
                    for l in range(0, B):
                        for m in range(0, 4):
                            temp_ar[i][j][k][l][m] = np.concatenate(
                                (zero_ar, origin_ar[i][j][k][l][m], origin_ar[i][j][k][l][m], origin_ar[i][j][k][l][m]),
                                axis=0)
        # concat 4 in_channel
        filter_ar = np.repeat(zero, D).reshape(A, 2, 4, B, 9, 4)
        for i in range(0, A):
            for j in range(0, 2):
                for k in range(0, 4):
                    for l in range(0, B):
                        filter_ar[i][j][k][l] = temp_ar[i][j][k][l].T
        # print('aaaa')
    elif kenel_size == 9:
        origin_ar = origin_ar.reshape(A, 2, 4, B, 4, 9)  # Using "2" make 8 outch
        zero = '0000'
        # concat 4 in channel
        filter_ar = np.repeat(zero, origin_size).reshape(A, 2, 4, B, 9, 4)

        for fn in range(0, A):
            for fc2 in range(0, 2):
                for fc1 in range(0, 4):
                    for incn in range(0, B):
                        filter_ar[fn][fc2][fc1][incn] = origin_ar[fn][fc2][fc1][incn].T
        # print('ssss')

    # to concat 4 filter
    filter_ar2 = np.repeat(zero, D).reshape(A, 2, B, 9, 16)
    for fn in range(0, A):
        for fin in range(0, 2):
            for cn in range(0, B):
                for d in range(0, 9):
                    filter_ar2[fn][fin][cn][d] = np.concatenate(
                        (filter_ar[fn][fin][0][cn][d], filter_ar[fn][fin][1][cn][d],
                         filter_ar[fn][fin][2][cn][d],
                         filter_ar[fn][fin][3][cn][d]), axis=0)

    filter_ar3 = np.repeat(zero, D).reshape(A, B, 2, 9, 16)  # to concat filter twice
    filter_ar3 = filter_ar2.transpose(0, 2, 1, 3, 4)

    Wconvert_ar = filter_ar3.reshape(C, 16)

    space = np.repeat(zero, 1).reshape(1, 1)
    vector_array = np.repeat(zero, E).reshape(A, 2, 16)  # outchannel num/8,2,16

    for i in range(0, A):  # filter_num / 8
        vector_array[i][0] = np.concatenate(
            (bias_active_ar[i * 8], bias_active_ar[i * 8 + 1], bias_active_ar[i * 8 + 2],
             bias_active_ar[i * 8 + 3], batch_sub_ar[i * 8], batch_sub_ar[i * 8 + 1],
             batch_sub_ar[i * 8 + 2], batch_sub_ar[i * 8 + 3],
             bias_active_ar[i * 8 + 4], bias_active_ar[i * 8 + 5],
             bias_active_ar[i * 8 + 6], bias_active_ar[i * 8 + 7], batch_sub_ar[i * 8 + 4],
             batch_sub_ar[i * 8 + 5], batch_sub_ar[i * 8 + 6], batch_sub_ar[i * 8 + 7]),
            axis=0)
        vector_array[i][1] = np.concatenate((batch_mul_ar[i * 8], batch_mul_ar[i * 8 + 1], batch_mul_ar[i * 8 + 2],
                                             batch_mul_ar[i * 8 + 3], batch_add_ar[i * 8], batch_add_ar[i * 8 + 1],
                                             batch_add_ar[i * 8 + 2], batch_add_ar[i * 8 + 3],
                                             batch_mul_ar[i * 8 + 4], batch_mul_ar[i * 8 + 5], batch_mul_ar[i * 8 + 6],
                                             batch_mul_ar[i * 8 + 7], batch_add_ar[i * 8 + 4], batch_add_ar[i * 8 + 5],
                                             batch_add_ar[i * 8 + 6], batch_add_ar[i * 8 + 7]), axis=0)

    vector_array = vector_array.reshape(F, 16)

    zero = '0000'

    Wconvert_array = Wconvert_ar.reshape(A, B, 18, 16)  # fil/8, in/4
    biacsc_ar = np.repeat(zero, E).reshape(A, 1, 2, 16)
    biacsc_ar = np.array(vector_array).reshape(A, 1, 2, 16)
    concat_ar = np.repeat(zero, G).reshape(int(A / 2), B * 2, 20, 16)

    for k in range(0, int(A / 2)):
        for i in range(0, B):
            concat_ar[k][2 * i] = np.concatenate((Wconvert_array[2 * k][i], biacsc_ar[2 * k][0]), axis=0)
            concat_ar[k][2 * i + 1] = np.concatenate((Wconvert_array[2 * k + 1][i], biacsc_ar[2 * k + 1][0]), axis=0)

    convert_ar = np.repeat(zero, G).reshape(H, 16)
    convert_ar = concat_ar.reshape(H, 16)

    
    Weight_List = []
    Weight_List.clear()
    for value in convert_ar:
        Result = ''.join(value)
        Weight_List.append(Result)

    df = pd.DataFrame(Weight_List)
    # df.to_csv(Write_Path, index=False, header=False, sep='\t')
    df1, df2 = Separated_Weight_DDR_Channel(df)
    Weight_Layer0_Channel0 = df1.values.tolist()
    Weight_Layer0_Channel1 = df2.values.tolist()
    Weight_Layer0 = [Weight_Layer0_Channel0, Weight_Layer0_Channel1]
    return Weight_Layer0
'''

# YOLOv2 Training Ordering: Forward
# def New_Weight_Hardware_ReOrdering_Layer0(Filter_Num, In_Channel_Num, Data_List, Sub_List, Mul_List, Add_List, Iteration):
#     # ---------------------------- Filter_Num, In_Channel_Num should be changed -----------------------
#     # Filter_Num = 128
#     # In_Channel_Num = 1024
#     Filter_Num = Filter_Num
#     In_Channel_Num = In_Channel_Num

#     A = int(Filter_Num / 8)
#     B = int(In_Channel_Num / 4)
#     C = int((Filter_Num * In_Channel_Num * 9) / 16)
#     D = int(Filter_Num * In_Channel_Num * 9)

#     E = int(Filter_Num * 4)
#     F = int(Filter_Num / 4)
#     G = int(A * B * 20 * 16)
#     H = int(A * B * 20)
#     zero = '0000'

#     origin = pd.DataFrame(Data_List)
#     # -------------------------------------------------------------------------------
#     # --------------------------------Activation and BatchNorm-----------------------
#     batch_sub = pd.DataFrame(Sub_List)
#     batch_mul = pd.DataFrame(Mul_List)
#     batch_add = pd.DataFrame(Add_List)
#     if Iteration == "1":
#         bias_active_ar = np.repeat('0000', Filter_Num).reshape(Filter_Num, 1)  # outchannel num == filter num
#     else:
#         bias_active_ar = np.repeat('4120', Filter_Num).reshape(Filter_Num, 1)  # outchannel num == filter num 
#     batch_sub_ar = np.repeat(zero, Filter_Num).reshape(Filter_Num, 1)
#     batch_mul_ar = np.repeat(zero, Filter_Num).reshape(Filter_Num, 1)
#     batch_add_ar = np.repeat(zero, Filter_Num).reshape(Filter_Num, 1)

#     batch_sub_ar = np.array(batch_sub)
#     batch_mul_ar = np.array(batch_mul)
#     batch_add_ar = np.array(batch_add)

#     # -------------------------------------------------------------------------------
#     # ------------------------------- For Layer 0 ------------------------------------
#     origin_ar = np.array(origin)
#     origin_ar = origin_ar.reshape(16, 3, 9)

#     zero = '0000'
#     zero_ar = np.repeat(zero, 1872)  # 16*13*9
#     zero_ar = zero_ar.reshape(16, 13, 9)

#     temp = np.repeat(zero, 2304)  # 16*16*9
#     temp = temp.reshape(16, 16, 9)
#     for i in range(0, 16):
#         temp[i] = np.concatenate((origin_ar[i], zero_ar[i]), axis=0)

#     origin_ar = temp
#     # --------------------------------------------------------------------------------
#     # --------------------------------------------------------------------------------

#     origin_size = np.size(origin_ar)
#     # print(origin_size)
#     # kenel_size = int(origin_size / (Filter_Num * In_Channel_Num))
#     # print(kenel_size)


#     origin_ar = origin_ar.reshape(A, 2, 4, B, 4, 9)  # Using "2" make 8 outch
#     zero = '0000'
#     # concat 4 in channel
#     filter_ar = np.repeat(zero, origin_size).reshape(A, 2, 4, B, 9, 4)

#     filter_ar = origin_ar.transpose(0,1,2,3,5,4)

#     # to concat 4 filter
#     filter_ar2 = np.repeat(zero, D).reshape(A, 2, B, 9, 16)
                    
#     filter_ar_0 = filter_ar[:,:,0,:,:,:]
#     filter_ar_1 = filter_ar[:,:,1,:,:,:]
#     filter_ar_2 = filter_ar[:,:,2,:,:,:]
#     filter_ar_3 = filter_ar[:,:,3,:,:,:]

#     filter_ar2 = np.concatenate((filter_ar_0, filter_ar_1, filter_ar_2, filter_ar_3), axis= 4)


#     filter_ar3 = np.repeat(zero, D).reshape(A, B, 2, 9, 16)  # to concat filter twice
#     filter_ar3 = filter_ar2.transpose(0, 2, 1, 3, 4)

#     # Wconvert_ar = filter_ar3.reshape(C, 16)

#     vector_array = np.repeat(zero, E).reshape(A, 2, 16)  # outchannel num/8,2,16

#     for i in range(0, A):  # filter_num / 8
#         vector_array[i][0] = np.concatenate(
#             (bias_active_ar[i * 8], bias_active_ar[i * 8 + 1], bias_active_ar[i * 8 + 2],
#              bias_active_ar[i * 8 + 3], batch_sub_ar[i * 8], batch_sub_ar[i * 8 + 1],
#              batch_sub_ar[i * 8 + 2], batch_sub_ar[i * 8 + 3],
#              bias_active_ar[i * 8 + 4], bias_active_ar[i * 8 + 5],
#              bias_active_ar[i * 8 + 6], bias_active_ar[i * 8 + 7], batch_sub_ar[i * 8 + 4],
#              batch_sub_ar[i * 8 + 5], batch_sub_ar[i * 8 + 6], batch_sub_ar[i * 8 + 7]),
#             axis=0)
#         vector_array[i][1] = np.concatenate((batch_mul_ar[i * 8], batch_mul_ar[i * 8 + 1], batch_mul_ar[i * 8 + 2],
#                                              batch_mul_ar[i * 8 + 3], batch_add_ar[i * 8], batch_add_ar[i * 8 + 1],
#                                              batch_add_ar[i * 8 + 2], batch_add_ar[i * 8 + 3],
#                                              batch_mul_ar[i * 8 + 4], batch_mul_ar[i * 8 + 5], batch_mul_ar[i * 8 + 6],
#                                              batch_mul_ar[i * 8 + 7], batch_add_ar[i * 8 + 4], batch_add_ar[i * 8 + 5],
#                                              batch_add_ar[i * 8 + 6], batch_add_ar[i * 8 + 7]), axis=0)

#     vector_array = vector_array.reshape(A, 1, 2, 16)

#     zero = '0000'

#     Wconvert_array = filter_ar3.reshape(A, B, 18, 16)  # fil/8, in/4
#     # biacsc_ar = np.repeat(zero, E).reshape(A, 1, 2, 16)
#     # biacsc_ar = np.array(vector_array).reshape(A, 1, 2, 16)
#     concat_ar = np.repeat(zero, G).reshape(int(A / 2), B * 2, 20, 16)


#     biacsc_ar1 = np.repeat(zero,A*B*2*16).reshape(A,B,2,16)
#     biacsc_ar1 = np.repeat(vector_array,B, axis=1)
#     concat_ar1 = np.concatenate((Wconvert_array, biacsc_ar1), axis=2)
#     for k in range (0,int(A/2)):
#         for i in range (0,B):
#             concat_ar[k][2*i]   = concat_ar1[2*k][i]
#             concat_ar[k][2*i+1] = concat_ar1[2*k+1][i]

#     concat_ar = concat_ar.reshape(H, 16)

    
#     Weight_List = []
#     Weight_List.clear()
#     for value in concat_ar:
#         Result = ''.join(value)
#         Weight_List.append(Result)

#     df = pd.DataFrame(Weight_List)
#     # df.to_csv(Write_Path, index=False, header=False, sep='\t')
#     df1, df2 = Separated_Weight_DDR_Channel(df)
#     Weight_Layer0_Channel0 = df1.values.tolist()
#     Weight_Layer0_Channel1 = df2.values.tolist()
#     Weight_Layer0 = [Weight_Layer0_Channel0, Weight_Layer0_Channel1]

#     return Weight_Layer0



# # YOLOv2 Training Ordering: Forward
# def New_Weight_Hardware_ReOrdering_OtherLayer(Filter_Num, In_Channel_Num, Data_List, Sub_List, Mul_List, Add_List, Iteration):
#     # ---------------------------- Filter_Num, In_Channel_Num should be changed -----------------------
#     # Filter_Num = 128
#     # In_Channel_Num =1024
#     Filter_Num = Filter_Num
#     In_Channel_Num = In_Channel_Num

#     A = int(Filter_Num / 8)
#     B = int(In_Channel_Num / 4)
#     C = int((Filter_Num * In_Channel_Num * 9) / 16)
#     D = int(Filter_Num * In_Channel_Num * 9)

#     E = int(Filter_Num * 4)
#     F = int(Filter_Num / 4)
#     G = int(A * B * 20 * 16)
#     H = int(A * B * 20)
#     zero = '0000'
#     # ------------------------------ All files require a garbage value in the first line. (ex) Add "test" to the
#     # first line---------------------------

#     origin = pd.DataFrame(Data_List)
#     # -------------------------------------------------------------------------------
#     # --------------------------------Activation and BatchNorm-----------------------
#     batch_sub = pd.DataFrame(Sub_List)
#     batch_mul = pd.DataFrame(Mul_List)
#     batch_add = pd.DataFrame(Add_List)
#     if Iteration == "1":
#         bias_active_ar = np.repeat('0000', Filter_Num).reshape(Filter_Num, 1)  # outchannel num == filter num
#     else: 
#         bias_active_ar = np.repeat('4120', Filter_Num).reshape(Filter_Num, 1)  # outchannel num == filter num
#     batch_sub_ar = np.repeat(zero, Filter_Num).reshape(Filter_Num, 1)
#     batch_mul_ar = np.repeat(zero, Filter_Num).reshape(Filter_Num, 1)
#     batch_add_ar = np.repeat(zero, Filter_Num).reshape(Filter_Num, 1)

#     batch_sub_ar = np.array(batch_sub)
#     batch_mul_ar = np.array(batch_mul)
#     batch_add_ar = np.array(batch_add)
#     # --------------------------------------------------------------------------------

#     # ------------------------------- For Other Layers -------------------------------
#     origin_ar = np.array(origin)
#     # --------------------------------------------------------------------------------

#     origin_size = np.size(origin_ar)
#     # print(origin_size)
#     kenel_size = int(origin_size / (Filter_Num * In_Channel_Num))
#     # print(kenel_size)

#     if kenel_size == 1:
#         origin_ar = origin_ar.reshape(A, 2, 4, B, 4, 1)
#         zero = '0000'
#         zero_ar = np.repeat(zero, 6)
#         zero_ar = zero_ar.reshape(6)
#         temp_ar = np.repeat(zero, D).reshape(A, 2, 4, B, 4, 9)

#         temp_ar = np.concatenate( (zero_ar[:,:,:,:,:,:],origin_ar[:,:,:,:,:,:],origin_ar[:,:,:,:,:,:],origin_ar[:,:,:,:,:,:]), axis = 5)
#         # concat 4 in_channel
#         filter_ar = np.repeat(zero, D).reshape(A, 2, 4, B, 9, 4)
#         filter_ar = temp_ar.transpose(0,1,2,3,5,4)

#     elif kenel_size == 9:
#         origin_ar = origin_ar.reshape(A, 2, 4, B, 4, 9)  # Using "2" make 8 outch
#         zero = '0000'
#         # concat 4 in channel
#         filter_ar = np.repeat(zero, origin_size).reshape(A, 2, 4, B, 9, 4)
#         filter_ar = origin_ar.transpose(0,1,2,3,5,4)

#     # to concat 4 filter
#     filter_ar2 = np.repeat(zero, D).reshape(A, 2, B, 9, 16)
#     filter_ar_0 = filter_ar[:,:,0,:,:,:]
#     filter_ar_1 = filter_ar[:,:,1,:,:,:]
#     filter_ar_2 = filter_ar[:,:,2,:,:,:]
#     filter_ar_3 = filter_ar[:,:,3,:,:,:]

#     filter_ar2 = np.concatenate((filter_ar_0, filter_ar_1, filter_ar_2, filter_ar_3), axis= 4)


#     filter_ar3 = np.repeat(zero, D).reshape(A, B, 2, 9, 16)
#     filter_ar3 = filter_ar2.transpose(0, 2, 1, 3, 4)

#     # Wconvert_ar = filter_ar3.reshape(C, 16)

#     vector_array = np.repeat(zero, E).reshape(A, 2, 16)

#     for i in range(0, A):  # filter_num / 8
#         vector_array[i][0] = np.concatenate((
#             bias_active_ar[i * 8], bias_active_ar[i * 8 + 1], bias_active_ar[i * 8 + 2],
#             bias_active_ar[i * 8 + 3], batch_sub_ar[i * 8], batch_sub_ar[i * 8 + 1],
#             batch_sub_ar[i * 8 + 2], batch_sub_ar[i * 8 + 3],
#             bias_active_ar[i * 8 + 4], bias_active_ar[i * 8 + 5],
#             bias_active_ar[i * 8 + 6], bias_active_ar[i * 8 + 7],
#             batch_sub_ar[i * 8 + 4], batch_sub_ar[i * 8 + 5], batch_sub_ar[i * 8 + 6],
#             batch_sub_ar[i * 8 + 7]), axis=0)
#         vector_array[i][1] = np.concatenate((batch_mul_ar[i * 8], batch_mul_ar[i * 8 + 1], batch_mul_ar[i * 8 + 2],
#                                              batch_mul_ar[i * 8 + 3], batch_add_ar[i * 8], batch_add_ar[i * 8 + 1],
#                                              batch_add_ar[i * 8 + 2], batch_add_ar[i * 8 + 3],
#                                              batch_mul_ar[i * 8 + 4], batch_mul_ar[i * 8 + 5], batch_mul_ar[i * 8 + 6],
#                                              batch_mul_ar[i * 8 + 7], batch_add_ar[i * 8 + 4], batch_add_ar[i * 8 + 5],
#                                              batch_add_ar[i * 8 + 6], batch_add_ar[i * 8 + 7]), axis=0)

#     vector_array = vector_array.reshape(A, 1, 2, 16)
#     zero = '0000'

#     Wconvert_array = filter_ar3.reshape(A, B, 18, 16)
#     # biacsc_ar = np.repeat(zero, E).reshape(A, 1, 2, 16)
#     # biacsc_ar = np.array(vector_array).reshape(A, 1, 2, 16)
#     concat_ar = np.repeat(zero, G).reshape(int(A / 2), B * 2, 20, 16)

#     biacsc_ar1 = np.repeat(zero,A*B*2*16).reshape(A,B,2,16)
#     biacsc_ar1 = np.repeat(vector_array,B, axis=1)
#     concat_ar1 = np.concatenate((Wconvert_array, biacsc_ar1), axis=2)
#     for k in range (0,int(A/2)):
#         for i in range (0,B):
#             concat_ar[k][2*i]   = concat_ar1[2*k][i]
#             concat_ar[k][2*i+1] = concat_ar1[2*k+1][i]


#     concat_ar = concat_ar.reshape(H, 16)

    
#     Weight_List = []
#     Weight_List.clear()
#     for value in concat_ar:
#         Result = ''.join(value)
#         Weight_List.append(Result)

#     df = pd.DataFrame(Weight_List)
#     # df.to_csv(Write_Path, index=False, header=False, sep='\t')
#     df1, df2 = Separated_Weight_DDR_Channel(df)
#     # df1.to_csv(Write_Path_Ch1, index=False, header=False, sep='\t')
#     # df2.to_csv(Write_Path_Ch2, index=False, header=False, sep='\t')
#     Weight_Layer1_7_Channel0 = df1.values.tolist()
#     Weight_Layer1_7_Channel1 = df2.values.tolist()
#     Weight_Layer1_7 = [Weight_Layer1_7_Channel0, Weight_Layer1_7_Channel1]
#     return Weight_Layer1_7


# # YOLOv2 Training Ordering: Forward
# def New_Weight_Hardware_ReOrdering_Layer8(Filter_Num, In_Channel_Num, Data_List, Bias_List):
#     # Filter_Num = 128
#     # In_Channel_Num = 1024
#     Filter_Num = Filter_Num
#     In_Channel_Num = In_Channel_Num

#     A = int(Filter_Num / 8)
#     B = int(In_Channel_Num / 4)
#     C = int((Filter_Num * In_Channel_Num * 9) / 16)
#     D = int(Filter_Num * In_Channel_Num * 9)

#     E = int(Filter_Num * 4)
#     F = int(Filter_Num / 4)
#     G = int(A * B * 20 * 16)
#     H = int(A * B * 20)
#     # zero = '0000'
    
#     # Padding the Weight List
#     Weight_Length = 131072
#     padding_size = 3072
#     current_length = len(Data_List)
#     padding_needed = max(0, Weight_Length - current_length)
#     padding_count = min(padding_needed, padding_size)
#     Data_List_ = Data_List + ["0000"] * padding_count

#     # Padding the Bias List
#     Bias_Length = 128
#     padding_size = 3
#     current_length = len(Bias_List)
#     padding_needed = max(0, Bias_Length - current_length)
#     padding_count = min(padding_needed, padding_size)
#     Bias_List_ = Bias_List + ["0000"] * padding_count

#     origin = pd.DataFrame(Data_List_)

#     # ------------------------------- Only Bias ----------------------------------------
#     bias = pd.DataFrame(Bias_List_)
#     # ------------------------------- For Other Layers -------------------------------
#     origin_ar = np.array(origin)
#     bias_active_ar = np.array(bias)
#     # --------------------------------------------------------------------------------

#     origin_size = np.size(origin_ar)
#     kenel_size = int(origin_size / (Filter_Num * In_Channel_Num))

#     if (kenel_size == 1):
#         origin_ar = origin_ar.reshape(A, 2, 4, B, 4, 1)
#         zero = '0000'
#         zero_ar = np.repeat(zero, 6)
#         zero_ar = zero_ar.reshape(6)
#         temp_ar = np.repeat(zero, D).reshape(A, 2, 4, B, 4, 9)
#         for i in range(0, A):
#             for j in range(0, 2):
#                 for k in range(0, 4):
#                     for l in range(0, B):
#                         for m in range(0, 4):
#                             temp_ar[i][j][k][l][m] = np.concatenate(
#                                 (zero_ar, origin_ar[i][j][k][l][m], origin_ar[i][j][k][l][m], origin_ar[i][j][k][l][m]),
#                                 axis=0)
#         # concat 4 in_channel
#         filter_ar = np.repeat(zero, D).reshape(A, 2, 4, B, 9, 4)
#         for i in range(0, A):
#             for j in range(0, 2):
#                 for k in range(0, 4):
#                     for l in range(0, B):
#                         filter_ar[i][j][k][l] = temp_ar[i][j][k][l].T
#         # print('aaaa')
#     elif (kenel_size == 9):
#         origin_ar = origin_ar.reshape(A, 2, 4, B, 4, 9)  # Using "2" make 8 outch
#         zero = '0000'
#         # concat 4 in channel
#         filter_ar = np.repeat(zero, origin_size).reshape(A, 2, 4, B, 9, 4)

#         for fn in range(0, A):
#             for fc2 in range(0, 2):
#                 for fc1 in range(0, 4):
#                     for incn in range(0, B):
#                         filter_ar[fn][fc2][fc1][incn] = origin_ar[fn][fc2][fc1][incn].T
#         # print('ssss')

#     # to concat 4 filter
#     filter_ar2 = np.repeat(zero, D).reshape(A, 2, B, 9, 16)
#     for fn in range(0, A):
#         for fin in range(0, 2):
#             for cn in range(0, B):
#                 for d in range(0, 9):
#                     filter_ar2[fn][fin][cn][d] = np.concatenate(
#                         (filter_ar[fn][fin][0][cn][d], filter_ar[fn][fin][1][cn][d],
#                          filter_ar[fn][fin][2][cn][d],
#                          filter_ar[fn][fin][3][cn][d]), axis=0)

#     filter_ar3 = np.repeat(zero, D).reshape(A, B, 2, 9, 16)  # to concat filter twice
#     filter_ar3 = filter_ar2.transpose(0, 2, 1, 3, 4)

#     Wconvert_ar = filter_ar3.reshape(C, 16)

#     space = np.repeat(zero, 1).reshape(1, 1)
#     vector_array = np.repeat(zero, E).reshape(A, 2, 16)  # outchannel num/8,2,16

#     for i in range(0, A):  # filter_num / 8
#         vector_array[i][0] = np.concatenate(
#             (bias_active_ar[i * 8], bias_active_ar[i * 8 + 1], bias_active_ar[i * 8 + 2],
#              bias_active_ar[i * 8 + 3], space[0], space[0],
#              space[0], space[0],
#              bias_active_ar[i * 8 + 4], bias_active_ar[i * 8 + 5],
#              bias_active_ar[i * 8 + 6], bias_active_ar[i * 8 + 7], space[0],
#              space[0], space[0], space[0]),
#             axis=0)
#         vector_array[i][1] = np.concatenate((space[0], space[0], space[0], space[0],
#                                              space[0], space[0], space[0], space[0],
#                                              space[0], space[0], space[0], space[0],
#                                              space[0], space[0], space[0], space[0]), axis=0)

#     vector_array = vector_array.reshape(F, 16)

#     zero = '0000'

#     Wconvert_array = Wconvert_ar.reshape(A, B, 18, 16)  # fil/8, in/4
#     biacsc_ar = np.repeat(zero, E).reshape(A, 1, 2, 16)
#     biacsc_ar = np.array(vector_array).reshape(A, 1, 2, 16)
#     concat_ar = np.repeat(zero, G).reshape(int(A / 2), B * 2, 20, 16)

#     for k in range(0, int(A / 2)):
#         for i in range(0, B):
#             concat_ar[k][2 * i] = np.concatenate((Wconvert_array[2 * k][i], biacsc_ar[2 * k][0]), axis=0)
#             concat_ar[k][2 * i + 1] = np.concatenate((Wconvert_array[2 * k + 1][i], biacsc_ar[2 * k + 1][0]), axis=0)

#     convert_ar = np.repeat(zero, G).reshape(H, 16)
#     convert_ar = concat_ar.reshape(H, 16)

    
#     Weight_List = []
#     Weight_List.clear()
#     for value in convert_ar:
#         Result = ''.join(value)
#         Weight_List.append(Result)
#     df = pd.DataFrame(Weight_List)
#     df1, df2 = Separated_Weight_DDR_Channel(df)
#     Weight_Layer8_Channel0 = df1.values.tolist()
#     Weight_Layer8_Channel1 = df2.values.tolist()
#     Weight_Layer8 = [Weight_Layer8_Channel0, Weight_Layer8_Channel1]
#     return Weight_Layer8



def New_Weight_Hardware_ReOrdering_Layer0(Filter_Num, In_Channel_Num, Data_List, Sub_List, Mul_List, Add_List, Iteration):

    zero = '0000'
    origin = pd.DataFrame(Data_List)
    batch_sub = pd.DataFrame(Sub_List)
    batch_mul = pd.DataFrame(Mul_List)
    batch_add = pd.DataFrame(Add_List)
    if Iteration == "1":
        bias_active_ar = np.repeat('0000', Filter_Num).reshape(Filter_Num, 1)  # outchannel num == filter num
    else:
        bias_active_ar = np.repeat('4120', Filter_Num).reshape(Filter_Num, 1)  # outchannel num == filter num 

    batch_sub_ar = np.array(batch_sub)
    batch_mul_ar = np.array(batch_mul)
    batch_add_ar = np.array(batch_add)

    # -------------------------------------------------------------------------------
    # ------------------------------- For Layer 0 ------------------------------------
    origin_ar = np.array(origin)
    origin_ar = origin_ar.reshape(16, 3, 9)

    zero_ar = np.repeat(zero, 1872).reshape(16, 13, 9)

    origin_ar = np.concatenate( (origin_ar,zero_ar), axis=1 )
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    origin_size = np.size(origin_ar)
    kenel_size = int(origin_size/(Filter_Num*In_Channel_Num))

    vector_array = np.repeat(zero,Filter_Num*4).reshape(Filter_Num//8,2,16) #outchannel num/8,2,16

    for i in range(0,Filter_Num//8): # filter_num / 8
        vector_array[i][0] = np.concatenate( (bias_active_ar[i*8],bias_active_ar[i*8+1],bias_active_ar[i*8+2],bias_active_ar[i*8+3],   batch_sub_ar[i*8],batch_sub_ar[i*8+1],batch_sub_ar[i*8+2],batch_sub_ar[i*8+3],
                                                bias_active_ar[i*8+4],bias_active_ar[i*8+5],bias_active_ar[i*8+6],bias_active_ar[i*8+7], batch_sub_ar[i*8+4],batch_sub_ar[i*8+5],batch_sub_ar[i*8+6],batch_sub_ar[i*8+7] ),axis=0)
        vector_array[i][1] = np.concatenate( (batch_mul_ar[i*8],batch_mul_ar[i*8+1],batch_mul_ar[i*8+2],batch_mul_ar[i*8+3],   batch_add_ar[i*8],batch_add_ar[i*8+1],batch_add_ar[i*8+2],batch_add_ar[i*8+3],
                                                batch_mul_ar[i*8+4],batch_mul_ar[i*8+5],batch_mul_ar[i*8+6],batch_mul_ar[i*8+7], batch_add_ar[i*8+4],batch_add_ar[i*8+5],batch_add_ar[i*8+6],batch_add_ar[i*8+7] ),axis=0)

    vector_array = vector_array.reshape(Filter_Num//8,1,2,16)

    biacsc_ar = np.repeat(vector_array,In_Channel_Num//8, axis=1).reshape(Filter_Num//8,In_Channel_Num//16,2,2,16) #(Filter_Num//8, In_Channel_Num//8, 2, 16)

    concat_ar = origin_ar.reshape(Filter_Num//8,8,In_Channel_Num//16,16,kenel_size)

    concat_ar = concat_ar.transpose(0,2,1,3,4).reshape(Filter_Num//8,In_Channel_Num//16,8,16,9)


    in0to7 = np.concatenate( (concat_ar[:,:,:,0:4], concat_ar[:,:,:,4:8]), axis=2 ).reshape(Filter_Num*In_Channel_Num//128,4,16,9)  #(Filter_Num//8,In_Channel_Num//16,16,4,kenel_size)
    in8to15 = np.concatenate( (concat_ar[:,:,:,8:12], concat_ar[:,:,:,12:16]), axis=2 ).reshape(Filter_Num*In_Channel_Num//128,4,16,9)

    in0to7 = in0to7.transpose(0,1,3,2).reshape(Filter_Num//8,In_Channel_Num//16,2,9*2,16)  #(Filter_Num//8,In_Channel_Num//16,4,kenel_size,16)
    in8to15 = in8to15.transpose(0,1,3,2).reshape(Filter_Num//8,In_Channel_Num//16,2,9*2,16)

    final_ar1 = np.concatenate( (in0to7,biacsc_ar), axis=3 ).reshape(Filter_Num//16,2,In_Channel_Num//16,2,20,16) #(Filter_Num//8,In_Channel_Num//16,2,20,16)
    final_ar2 = np.concatenate( (in8to15,biacsc_ar), axis=3 ).reshape(Filter_Num//16,2,In_Channel_Num//16,2,20,16)

    final_ar1 = final_ar1.transpose(0,2,3,1,4,5).reshape(Filter_Num*In_Channel_Num*5//16,16)
    final_ar2 = final_ar2.transpose(0,2,3,1,4,5).reshape(Filter_Num*In_Channel_Num*5//16,16)

    final_ar = np.concatenate((final_ar1,final_ar2), axis=0)

    final_list = []
    final_list.clear()
    for value in final_ar:
        Result = ''.join(value)
        final_list.append(Result)
    df = pd.DataFrame(final_list)
    final_ar = np.array(df).reshape(2,Filter_Num*In_Channel_Num*5//16)
    df1 = pd.DataFrame(final_ar[0])
    df2 = pd.DataFrame(final_ar[1])

    Weight_Channel0 = df1.values.tolist()
    Weight_Channel1 = df2.values.tolist()
    Weight_Layer0 = [Weight_Channel0, Weight_Channel1]

    return Weight_Layer0

# YOLOv2 Training Ordering: Forward
def New_Weight_Hardware_ReOrdering_OtherLayer(Filter_Num, In_Channel_Num, Data_List, Sub_List, Mul_List, Add_List, Iteration):
 
    zero = '0000'

    origin = pd.DataFrame(Data_List)
    batch_sub = pd.DataFrame(Sub_List)
    batch_mul = pd.DataFrame(Mul_List)
    batch_add = pd.DataFrame(Add_List)
    if Iteration == "1":
        bias_active_ar = np.repeat('0000', Filter_Num).reshape(Filter_Num, 1)  # outchannel num == filter num
    else: 
        bias_active_ar = np.repeat('4120', Filter_Num).reshape(Filter_Num, 1)  # outchannel num == filter num

    batch_sub_ar = np.array(batch_sub)
    batch_mul_ar = np.array(batch_mul)
    batch_add_ar = np.array(batch_add)
    origin_ar = np.array(origin)
    # --------------------------------------------------------------------------------

    origin_size = np.size(origin_ar)
    kenel_size = int(origin_size/(Filter_Num*In_Channel_Num))

    vector_array = np.repeat(zero,Filter_Num*4).reshape(Filter_Num//8,2,16) #outchannel num/8,2,16

    for i in range(0,Filter_Num//8): # filter_num / 8
        vector_array[i][0] = np.concatenate( (bias_active_ar[i*8],bias_active_ar[i*8+1],bias_active_ar[i*8+2],bias_active_ar[i*8+3],   batch_sub_ar[i*8],batch_sub_ar[i*8+1],batch_sub_ar[i*8+2],batch_sub_ar[i*8+3],
                                                bias_active_ar[i*8+4],bias_active_ar[i*8+5],bias_active_ar[i*8+6],bias_active_ar[i*8+7], batch_sub_ar[i*8+4],batch_sub_ar[i*8+5],batch_sub_ar[i*8+6],batch_sub_ar[i*8+7] ),axis=0)
        vector_array[i][1] = np.concatenate( (batch_mul_ar[i*8],batch_mul_ar[i*8+1],batch_mul_ar[i*8+2],batch_mul_ar[i*8+3],   batch_add_ar[i*8],batch_add_ar[i*8+1],batch_add_ar[i*8+2],batch_add_ar[i*8+3],
                                                batch_mul_ar[i*8+4],batch_mul_ar[i*8+5],batch_mul_ar[i*8+6],batch_mul_ar[i*8+7], batch_add_ar[i*8+4],batch_add_ar[i*8+5],batch_add_ar[i*8+6],batch_add_ar[i*8+7] ),axis=0)

    vector_array = vector_array.reshape(Filter_Num//8,1,2,16)

    biacsc_ar = np.repeat(vector_array,In_Channel_Num//8, axis=1).reshape(Filter_Num//8,In_Channel_Num//16,2,2,16) #(Filter_Num//8, In_Channel_Num//8, 2, 16)

    concat_ar = origin_ar.reshape(Filter_Num//8,8,In_Channel_Num//16,16,kenel_size)

    concat_ar = concat_ar.transpose(0,2,1,3,4).reshape(Filter_Num//8,In_Channel_Num//16,8,16,9)


    in0to7 = np.concatenate( (concat_ar[:,:,:,0:4], concat_ar[:,:,:,4:8]), axis=2 ).reshape(Filter_Num*In_Channel_Num//128,4,16,9)  #(Filter_Num//8,In_Channel_Num//16,16,4,kenel_size)
    in8to15 = np.concatenate( (concat_ar[:,:,:,8:12], concat_ar[:,:,:,12:16]), axis=2 ).reshape(Filter_Num*In_Channel_Num//128,4,16,9)

    in0to7 = in0to7.transpose(0,1,3,2).reshape(Filter_Num//8,In_Channel_Num//16,2,9*2,16)  #(Filter_Num//8,In_Channel_Num//16,4,kenel_size,16)
    in8to15 = in8to15.transpose(0,1,3,2).reshape(Filter_Num//8,In_Channel_Num//16,2,9*2,16)

    final_ar1 = np.concatenate( (in0to7,biacsc_ar), axis=3 ).reshape(Filter_Num//16,2,In_Channel_Num//16,2,20,16) #(Filter_Num//8,In_Channel_Num//16,2,20,16)
    final_ar2 = np.concatenate( (in8to15,biacsc_ar), axis=3 ).reshape(Filter_Num//16,2,In_Channel_Num//16,2,20,16)

    final_ar1 = final_ar1.transpose(0,2,3,1,4,5).reshape(Filter_Num*In_Channel_Num*5//16,16)
    final_ar2 = final_ar2.transpose(0,2,3,1,4,5).reshape(Filter_Num*In_Channel_Num*5//16,16)

    final_ar = np.concatenate((final_ar1,final_ar2), axis=0)

    final_list = []
    final_list.clear()
    for value in final_ar:
        Result = ''.join(value)
        final_list.append(Result)
    df = pd.DataFrame(final_list)
    final_ar = np.array(df).reshape(2,Filter_Num*In_Channel_Num*5//16)
    df1 = pd.DataFrame(final_ar[0])
    df2 = pd.DataFrame(final_ar[1])

    Weight_Channel0 = df1.values.tolist()
    Weight_Channel1 = df2.values.tolist()
    Weight_Layer1_7 = [Weight_Channel0, Weight_Channel1]

    return Weight_Layer1_7


# YOLOv2 Training Ordering: Forward
def New_Weight_Hardware_ReOrdering_Layer8(Filter_Num, In_Channel_Num, Data_List, Bias_List):

    zero = '0000'
    Data_List_ = Data_List + ["0000"] * 3072
    Bias_List_ = Bias_List + ["0000"] * 3

    origin = pd.DataFrame(Data_List_)
    bias = pd.DataFrame(Bias_List_)

    origin_ar = np.array(origin)
    bias_active_ar = np.array(bias)
    batch_sub_ar = np.repeat(zero,Filter_Num).reshape(Filter_Num,1)
    batch_mul_ar = np.repeat(zero,Filter_Num).reshape(Filter_Num,1)
    batch_add_ar = np.repeat(zero,Filter_Num).reshape(Filter_Num,1)

    # --------------------------------------------------------------------------------

    origin_size = np.size(origin_ar)
    kenel_size = int(origin_size/(Filter_Num*In_Channel_Num))

    vector_array = np.repeat(zero,Filter_Num*4).reshape(Filter_Num//8,2,16) #outchannel num/8,2,16

    for i in range(0,Filter_Num//8): # filter_num / 8
        vector_array[i][0] = np.concatenate( (bias_active_ar[i*8],bias_active_ar[i*8+1],bias_active_ar[i*8+2],bias_active_ar[i*8+3],   batch_sub_ar[i*8],batch_sub_ar[i*8+1],batch_sub_ar[i*8+2],batch_sub_ar[i*8+3],
                                                bias_active_ar[i*8+4],bias_active_ar[i*8+5],bias_active_ar[i*8+6],bias_active_ar[i*8+7], batch_sub_ar[i*8+4],batch_sub_ar[i*8+5],batch_sub_ar[i*8+6],batch_sub_ar[i*8+7] ),axis=0)
        vector_array[i][1] = np.concatenate( (batch_mul_ar[i*8],batch_mul_ar[i*8+1],batch_mul_ar[i*8+2],batch_mul_ar[i*8+3],   batch_add_ar[i*8],batch_add_ar[i*8+1],batch_add_ar[i*8+2],batch_add_ar[i*8+3],
                                                batch_mul_ar[i*8+4],batch_mul_ar[i*8+5],batch_mul_ar[i*8+6],batch_mul_ar[i*8+7], batch_add_ar[i*8+4],batch_add_ar[i*8+5],batch_add_ar[i*8+6],batch_add_ar[i*8+7] ),axis=0)

    vector_array = vector_array.reshape(Filter_Num//8,1,2,16)

    biacsc_ar = np.repeat(vector_array,In_Channel_Num//8, axis=1).reshape(Filter_Num//8,In_Channel_Num//16,2,2,16) #(Filter_Num//8, In_Channel_Num//8, 2, 16)

    origin_ar = origin_ar.reshape(Filter_Num*In_Channel_Num,kenel_size)
    zero_ar = np.repeat(zero,Filter_Num*In_Channel_Num*6).reshape(Filter_Num*In_Channel_Num,6)
    concat_ar = np.concatenate((zero_ar, origin_ar, origin_ar, origin_ar), axis=1).reshape(Filter_Num//8,8,In_Channel_Num//16,16,9)

    concat_ar = concat_ar.transpose(0,2,1,3,4).reshape(Filter_Num//8,In_Channel_Num//16,8,16,9)

    in0to7 = np.concatenate( (concat_ar[:,:,:,0:4], concat_ar[:,:,:,4:8]), axis=2 ).reshape(Filter_Num*In_Channel_Num//128,4,16,9)  #(Filter_Num//8,In_Channel_Num//16,16,4,kenel_size)
    in8to15 = np.concatenate( (concat_ar[:,:,:,8:12], concat_ar[:,:,:,12:16]), axis=2 ).reshape(Filter_Num*In_Channel_Num//128,4,16,9)

    in0to7 = in0to7.transpose(0,1,3,2).reshape(Filter_Num//8,In_Channel_Num//16,2,9*2,16)  #(Filter_Num//8,In_Channel_Num//16,4,kenel_size,16)
    in8to15 = in8to15.transpose(0,1,3,2).reshape(Filter_Num//8,In_Channel_Num//16,2,9*2,16)

    final_ar1 = np.concatenate( (in0to7,biacsc_ar), axis=3 ).reshape(Filter_Num//16,2,In_Channel_Num//16,2,20,16) #(Filter_Num//8,In_Channel_Num//16,2,20,16)
    final_ar2 = np.concatenate( (in8to15,biacsc_ar), axis=3 ).reshape(Filter_Num//16,2,In_Channel_Num//16,2,20,16)

    final_ar1 = final_ar1.transpose(0,2,3,1,4,5).reshape(Filter_Num*In_Channel_Num*5//16,16)
    final_ar2 = final_ar2.transpose(0,2,3,1,4,5).reshape(Filter_Num*In_Channel_Num*5//16,16)

    final_ar = np.concatenate((final_ar1,final_ar2), axis=0)

    final_list = []
    final_list.clear()
    for value in final_ar:
        Result = ''.join(value)
        final_list.append(Result)
    df = pd.DataFrame(final_list)
    final_ar = np.array(df).reshape(2,Filter_Num*In_Channel_Num*5//16)
    df1 = pd.DataFrame(final_ar[0])
    df2 = pd.DataFrame(final_ar[1])

    Weight_Channel0 = df1.values.tolist()
    Weight_Channel1 = df2.values.tolist()
    Weight_Layer8 = [Weight_Channel0, Weight_Channel1]

    return Weight_Layer8








def ReLu_Marking_Ordering(Out_CH, Out_Size, DataCH0_List, DataCH1_List):
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

    # outlist = []
    outlist = final_ar.tolist()
        
    return outlist

def bfloat16_to_decimal(hex_str):
    # 32 비트 부동 소수점 값을 hex 문자열로 변환
    float32_hex = hex_str.ljust(8,'0')
    hex_data = bytes.fromhex(float32_hex)
    # hex 문자열을 부동 소수점 값으로 언팩
    decimal_value = struct.unpack('!f', hex_data)[0]

    return decimal_value
 
def Fmap_Reverse_Ordering(Out_CH, Out_Size, DataCH0_List, DataCH1_List):
    # Out_CH = 128
    # Out_Size = 13
    Out_CH = Out_CH
    Out_Size = Out_Size

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
    

    
    def to_decimal(num):
        return float(num)

    decimal_vectorized = np.vectorize(to_decimal)

    outlist = decimal_vectorized(final_ar)
    outlist = outlist.tolist()
        
    return outlist

'''
def Fmap_Reverse_Ordering(Out_CH, Out_Size, DataCH0_List, DataCH1_List):
    # Out_CH = 128
    # Out_Size = 13
    Out_CH = Out_CH
    Out_Size = Out_Size

    origin0 = pd.DataFrame(DataCH0_List)
    origin1 = pd.DataFrame(DataCH1_List)
    
    origin_ar0 = np.array(origin0)
    origin_ar1 = np.array(origin1)
    origin_size = np.size(origin_ar0)
    A = int(origin_size * 16)
    iter_13 = int(Out_Size / 13)
    
    convert_ar0 = np.repeat('0000', A).reshape(origin_size, 16)
    convert_ar1 = np.repeat('0000', A).reshape(origin_size, 16)

    def splitCount(s, count):
        return [''.join(x) for x in zip(*[list(s[z::count]) for z in range(count)])]

    for i in range(0, origin_size):
        convert_ar0[i] = splitCount(origin_ar0[i][0], 4)
        convert_ar1[i] = splitCount(origin_ar1[i][0], 4)

    concat_ar = np.repeat('0000',A*2).reshape(origin_size*2,16)

    # for i in range(0,int(origin_size/2)):
    #   concat_ar[4*i]   = convert_ar0[2*i]
    #   concat_ar[4*i+1] = convert_ar0[2*i+1]
    #   concat_ar[4*i+2] = convert_ar1[2*i]
    #   concat_ar[4*i+3] = convert_ar1[2*i+1]

    convert_ar0 = convert_ar0.reshape(origin_size//2, 2, 16)
    convert_ar1 = convert_ar1.reshape(origin_size//2, 2, 16)
    concat_ar = np.concatenate((convert_ar0, convert_ar1), axis=1)

    concat_ar = concat_ar.reshape(Out_CH//16, Out_Size*(Out_Size+iter_13*3), 16)

    # print("---{}s seconds 2---".format(time.time()-start_time))

    rearrangeCH_ar = np.repeat('0000',A*2).reshape(Out_CH//16, Out_Size*(Out_Size+iter_13*3), 16)

    # for m in range (0, int(Out_CH/16)):
    #   for l in range (0, int(Out_Size*(Out_Size+iter_13*3)/4)):
    #     for n in range (0,16):
    #       rearrangeCH_ar[m][l][n]                                        = concat_ar[m][4*l][n]
    #       rearrangeCH_ar[m][l+int(Out_Size*(Out_Size+iter_13*3)/4)][n]   = concat_ar[m][4*l+1][n]
    #       rearrangeCH_ar[m][l+int(Out_Size*(Out_Size+iter_13*3)/4)*2][n] = concat_ar[m][4*l+2][n]
    #       rearrangeCH_ar[m][l+int(Out_Size*(Out_Size+iter_13*3)/4)*3][n] = concat_ar[m][4*l+3][n]

    concat_ar = concat_ar.reshape(Out_CH//16, ((Out_Size*(Out_Size+iter_13*3))//4), 4, 16)
    rearrangeCH_ar = np.concatenate((concat_ar[:,:,0,:], concat_ar[:,:,1,:], concat_ar[:,:,2,:], concat_ar[:,:,3,:]), axis=1)

    rearrangeCH_ar = rearrangeCH_ar.reshape(int(Out_CH/4), Out_Size*(Out_Size+iter_13*3), 4)

    # print("---{}s seconds 3---".format(time.time()-start_time))

    trans_ar = np.repeat('0000', A*2).reshape(Out_CH, Out_Size*(Out_Size+iter_13*3), 1)

    # for i in range (0, int(Out_CH/4)):
    #   for j in range (0, Out_Size*(Out_Size+iter_13*3)):
    #     trans_ar[4*i][j][0]   = rearrangeCH_ar[i][j][0]
    #     trans_ar[4*i+1][j][0] = rearrangeCH_ar[i][j][1]
    #     trans_ar[4*i+2][j][0] = rearrangeCH_ar[i][j][2]
    #     trans_ar[4*i+3][j][0] = rearrangeCH_ar[i][j][3]
    trans_ar = np.concatenate((rearrangeCH_ar[:,:,0], rearrangeCH_ar[:,:,1], rearrangeCH_ar[:,:,2], rearrangeCH_ar[:,:,3]), axis=1)


    trans_ar = trans_ar.reshape(Out_CH, int(Out_Size*(Out_Size+iter_13*3)/4), 4)


    # print("---{}s seconds 4---".format(time.time()-start_time))

    convert_ar = np.repeat('0000', A*2).reshape(Out_CH, (Out_Size+iter_13*3), Out_Size)

    # for i in range (0, Out_CH):
    #   for j in range (0, int((Out_Size+iter_13*3)/4)):
    #     for l in range (0, Out_Size):
    #       convert_ar[i][4*j][l]   = trans_ar[i][j*Out_Size+l][0]
    #       convert_ar[i][4*j+1][l] = trans_ar[i][j*Out_Size+l][1]
    #       convert_ar[i][4*j+2][l] = trans_ar[i][j*Out_Size+l][2]
    #       convert_ar[i][4*j+3][l] = trans_ar[i][j*Out_Size+l][3]


    trans_ar0 = trans_ar[:,:,0].reshape(Out_CH, (Out_Size+iter_13*3)//4, Out_Size)
    trans_ar1 = trans_ar[:,:,1].reshape(Out_CH, (Out_Size+iter_13*3)//4, Out_Size)
    trans_ar2 = trans_ar[:,:,2].reshape(Out_CH, (Out_Size+iter_13*3)//4, Out_Size)
    trans_ar3 = trans_ar[:,:,3].reshape(Out_CH, (Out_Size+iter_13*3)//4, Out_Size)
    convert_ar = np.concatenate((trans_ar0, trans_ar1, trans_ar2, trans_ar3), axis=2)



    convert_ar = convert_ar.reshape(Out_CH, (Out_Size+iter_13*3)//16, 16, Out_Size)

    # print("---{}s seconds 5---".format(time.time()-start_time))

    final_ar = np.repeat('0000', Out_CH*Out_Size*Out_Size).reshape(Out_CH, Out_Size//13, 13, Out_Size)
    # for i in range (0, Out_CH):
    #   for j in range (0, iter_13):
    #     for l in range (0, Out_Size):
    #       final_ar[i][j*13+0][l]   = convert_ar[i][j*16+0][l]
    #       final_ar[i][j*13+1][l]   = convert_ar[i][j*16+1][l]
    #       final_ar[i][j*13+2][l]   = convert_ar[i][j*16+2][l]
    #       final_ar[i][j*13+3][l]   = convert_ar[i][j*16+3][l]
    #       final_ar[i][j*13+4][l]   = convert_ar[i][j*16+4][l]
    #       final_ar[i][j*13+5][l]   = convert_ar[i][j*16+5][l]
    #       final_ar[i][j*13+6][l]   = convert_ar[i][j*16+6][l]
    #       final_ar[i][j*13+7][l]   = convert_ar[i][j*16+7][l]
    #       final_ar[i][j*13+8][l]   = convert_ar[i][j*16+8][l]
    #       final_ar[i][j*13+9][l]   = convert_ar[i][j*16+9][l]
    #       final_ar[i][j*13+10][l]  = convert_ar[i][j*16+10][l]
    #       final_ar[i][j*13+11][l]  = convert_ar[i][j*16+11][l]
    #       final_ar[i][j*13+12][l]  = convert_ar[i][j*16+12][l]

    final_ar = convert_ar[:,:,0:13,:]
    final_ar = final_ar.reshape(Out_CH, Out_Size, Out_Size)
    # print("---{}s seconds 6---".format(time.time()-start_time))

    # for i in range (0,Out_CH):
    #   final_ar[i] = final_ar[i].T

    final_ar = final_ar.transpose(0,2,1)

    final_ar = final_ar.reshape(Out_Size*Out_Size*Out_CH)
    
    
    OutputFmap_List = []
    OutputFmap_List.clear()
    for value in final_ar:
        Result = ''.join(value)
        OutputFmap_List.append(Result)
        
    return OutputFmap_List
'''

'''
def Weight_Gradient_Hardware_ReOrdering(Out_CH, In_CH, DataCH0_List, DataCH1_List):

    # Filter_Num = 128
    # In_Channel_Num = 1024
    Filter_Num = Out_CH
    In_Channel_Num = In_CH

    origin0 = pd.DataFrame(DataCH0_List)
    origin1 = pd.DataFrame(DataCH1_List)

    origin_ar0 = np.array(origin0)
    origin_ar1 = np.array(origin1)
    half_size = np.size(origin_ar0)
    A = int(half_size*16)
    kenel_size = int(half_size/(Filter_Num*In_Channel_Num/16/2))

    convert_ar0 = np.repeat('0000',A).reshape(half_size,16)
    convert_ar1 = np.repeat('0000',A).reshape(half_size,16)

    def splitCount(s, count):
        return [''.join(x) for x in zip(*[list(s[z::count]) for z in range(count)])]

    for i in range(0,half_size):
        convert_ar0[i] = splitCount(origin_ar0[i][0],4)
        convert_ar1[i] = splitCount(origin_ar1[i][0],4)

    origin_ar = np.repeat('0000',A*2).reshape(half_size*2,16)
    convert_ar = np.repeat('0000',A*2).reshape(half_size//(kenel_size*8),kenel_size*16,16)
    convert_ar0 = convert_ar0.reshape(half_size//(kenel_size*8),kenel_size*8,16)
    convert_ar1 = convert_ar1.reshape(half_size//(kenel_size*8),kenel_size*8,16)

    convert_ar = np.concatenate((convert_ar0, convert_ar1), axis=1)
    origin_ar = convert_ar.reshape(half_size*2,16)

    origin_size = np.size(origin_ar)
    zero = '0000'
    # print("---{}s seconds 2---".format(time.time()-start_time))
    origin_ar = origin_ar.reshape(Filter_Num//16,In_Channel_Num//2,kenel_size*2,16)
    Wconvert_array = np.repeat(zero, Filter_Num*In_Channel_Num*kenel_size).reshape(Filter_Num//8,In_Channel_Num//4,kenel_size*2,16)

    for k in range (0,int(Filter_Num/16)):
        for i in range (0,int(In_Channel_Num/4)):
            Wconvert_array[2*k][i]  = origin_ar[k][2*i]
            Wconvert_array[2*k+1][i]= origin_ar[k][2*i+1]



    Wconvert_array = Wconvert_array.reshape(Filter_Num//8,In_Channel_Num//4,2,kenel_size,16)


    # print("---{}s seconds 2---".format(time.time()-start_time))

    filter_ar2 = np.repeat(zero, Filter_Num*In_Channel_Num*kenel_size).reshape(Filter_Num//8,2,In_Channel_Num//4,kenel_size,16)
    filter_ar2 = Wconvert_array.transpose(0,2,1,3,4)

    filter_ar = np.repeat(zero, origin_size).reshape(Filter_Num//8, 2, 4, In_Channel_Num//4, kenel_size, 4)
    # for fn in range(0,int(Filter_Num/8)):
    #   for fin in range (0,2):
    #     for cn in range (0,int(In_Channel_Num/4)):
    #       for d in range (0,kenel_size):
    #         for ch in range (0,4):
    #           filter_ar[fn][fin][0][cn][d][ch] = filter_ar2[fn][fin][cn][d][ch]
    #           filter_ar[fn][fin][1][cn][d][ch] = filter_ar2[fn][fin][cn][d][ch+4]
    #           filter_ar[fn][fin][2][cn][d][ch] = filter_ar2[fn][fin][cn][d][ch+8]
    #           filter_ar[fn][fin][3][cn][d][ch] = filter_ar2[fn][fin][cn][d][ch+12]

    filter_ar[:, :, 0, :, :, :] = filter_ar2[:, :, :, :, 0:4]
    filter_ar[:, :, 1, :, :, :] = filter_ar2[:, :, :, :, 4:8]
    filter_ar[:, :, 2, :, :, :] = filter_ar2[:, :, :, :, 8:12]
    filter_ar[:, :, 3, :, :, :] = filter_ar2[:, :, :, :, 12:16]

    # print("---{}s seconds 3---".format(time.time()-start_time))

    convert_ar = np.repeat(zero, Filter_Num*In_Channel_Num*kenel_size).reshape(Filter_Num//8, 2, 4, In_Channel_Num//4, 4, kenel_size) # Using "2" make 8 outch
    # for fn in range (0,int(Filter_Num/8)):
    #   for fc2 in range (0,2):
    #     for fc1 in range (0,4):
    #       for incn in range (0,int(In_Channel_Num/4)):
    #           convert_ar[fn][fc2][fc1][incn] = filter_ar[fn][fc2][fc1][incn].T


    # Transpose and assign the values directly
    convert_ar[:, :, :, :, :, :] = filter_ar[:, :, :, :, :, :].transpose(0, 1, 2, 3, 5, 4)
    convert_ar = convert_ar.reshape(Filter_Num*In_Channel_Num*kenel_size)
  
    
    Weight_Gradient_List = []
    Weight_Gradient_List.clear()
    for value in convert_ar:
        Result = ''.join(value)
        Weight_Gradient_List.append(Result)
    #   df = pd.DataFrame(Weight_Gradient_List)
    #   df.to_csv(Write_Path,index=False, header=False, sep='\t')
    return Weight_Gradient_List
'''

# Hard2Software ReOrdering: Backward
# Weight Gradient ReOrdering: backward
def Weight_Gradient_Hardware_ReOrdering(Out_CH, In_CH, DataCH0_List, DataCH1_List):

    # Filter_Num = 128
    # In_Channel_Num = 1024
    Filter_Num = Out_CH
    In_Channel_Num = In_CH

    origin0 = pd.DataFrame(DataCH0_List)
    origin1 = pd.DataFrame(DataCH1_List)

    origin_ar0 = np.array(origin0)
    origin_ar1 = np.array(origin1)
    half_size = np.size(origin_ar0)//16
    kenel_size = int(half_size/(Filter_Num*In_Channel_Num/16/2))

    convert_ar0 = origin_ar0.reshape(half_size//kenel_size,kenel_size,16)
    convert_ar1 = origin_ar1.reshape(half_size//kenel_size,kenel_size,16)

    convert_ar0 = convert_ar0.transpose(0,2,1).reshape(Filter_Num*In_Channel_Num//256,2,16,4,kenel_size)
    convert_ar1 = convert_ar1.transpose(0,2,1).reshape(Filter_Num*In_Channel_Num//256,2,16,4,kenel_size)

    concat_ar = np.concatenate( (convert_ar0[:,0], convert_ar0[:,1], convert_ar1[:,0], convert_ar1[:,1]), axis=2 ) #shape(Filter_Num*In_Channel_Num//256,16,16,kenel_size)
    concat_ar = concat_ar.reshape(Filter_Num//16,In_Channel_Num//16,16,16,kenel_size)

    final_ar = concat_ar.transpose(0,2,1,3,4)
    final_ar = final_ar.reshape(-1)

    def to_decimal(num):

        return float(num)  

    decimal_vectorized = np.vectorize(to_decimal)

    outlist = decimal_vectorized(final_ar)

    outlist = outlist.tolist()
    return outlist


def Weight_Gradient_Hardware_ReOrdering_whole_image(Out_CH, In_CH, DataCH0_List, DataCH1_List):

    # Filter_Num = 128
    # In_Channel_Num = 1024
    Filter_Num = Out_CH
    In_Channel_Num = In_CH
    Batch_size = 8

    origin0 = pd.DataFrame(DataCH0_List)
    origin1 = pd.DataFrame(DataCH1_List)

    origin_ar0 = np.array(origin0)
    origin_ar1 = np.array(origin1)
    half_size = np.size(origin_ar0)//(16)
    kenel_size = int(half_size/(Batch_size*Filter_Num*In_Channel_Num/16/2))

    # convert_ar0 = np.repeat('0000',A).reshape(Batch_size*half_size,16)
    # convert_ar1 = np.repeat('0000',A).reshape(Batch_size*half_size,16)
    # origin_ar = np.repeat('0000',A*2).reshape(half_size*2,16)
    # convert_ar = np.repeat('0000',A*2).reshape(half_size//(kenel_size*8),kenel_size*16,16)

    convert_ar0 = origin_ar0.reshape(half_size//(kenel_size*8),kenel_size*8*16)
    convert_ar1 = origin_ar1.reshape(half_size//(kenel_size*8),kenel_size*8*16)

    origin_ar = np.concatenate((convert_ar0, convert_ar1), axis=1).reshape(Batch_size,Filter_Num//16,In_Channel_Num//2,kenel_size*2,16)

    origin_size = Batch_size*Filter_Num*In_Channel_Num*kenel_size
    zero = '0000'

    Wconvert_array = np.repeat(zero, origin_size).reshape(Batch_size,Filter_Num//8,In_Channel_Num//4,kenel_size*2,16)

    for b in range (0,Batch_size):
        for k in range (0,int(Filter_Num/16)):
            for i in range (0,int(In_Channel_Num/4)):
                Wconvert_array[b][2*k][i]  = origin_ar[b][k][2*i]
                Wconvert_array[b][2*k+1][i]= origin_ar[b][k][2*i+1]



    Wconvert_array = Wconvert_array.reshape(Batch_size,Filter_Num//8,In_Channel_Num//4,2,kenel_size,16)


    filter_ar2 = np.repeat(zero, origin_size).reshape(Batch_size,Filter_Num//8,2,In_Channel_Num//4,kenel_size,16)
    filter_ar2 = Wconvert_array.transpose(0,1,3,2,4,5)

    filter_ar = np.repeat(zero, origin_size).reshape(Batch_size, Filter_Num//8, 2, 4, In_Channel_Num//4, kenel_size, 4)


    filter_ar[:, :, :, 0, :, :, :] = filter_ar2[:, :, :, :, :, 0:4]
    filter_ar[:, :, :, 1, :, :, :] = filter_ar2[:, :, :, :, :, 4:8]
    filter_ar[:, :, :, 2, :, :, :] = filter_ar2[:, :, :, :, :, 8:12]
    filter_ar[:, :, :, 3, :, :, :] = filter_ar2[:, :, :, :, :, 12:16]

    convert_ar = np.repeat(zero, origin_size).reshape(Batch_size, Filter_Num//8, 2, 4, In_Channel_Num//4, 4, kenel_size) # Using "2" make 8 outch



    # Transpose and assign the values directly
    convert_ar = filter_ar.transpose(0, 1, 2, 3, 4, 6, 5)
    convert_ar = convert_ar.reshape(origin_size)
  
    
    '''
    # # bfloat16_array = np.array(origin_ar, dtype=np.uint16)  # bfloat16 데이터
    # decimal_array = np.vectorize(bfloat16_to_decimal)(convert_ar).reshape(Batch_size, origin_size//Batch_size)

    # def bfloat16_to_decimal(bfloat16_data):
    #     int_data = int(bfloat16_data, 16)
    #     sign = 1 if (int_data & 0x8000) == 0 else -1
    #     exponent = ((int_data & 0x7F80) >> 7) - 127
    #     fraction = 1.0 + (int_data & 0x007F) / 128.0
    #     if(int_data != 0):
    #         return sign * fraction * (2 ** exponent)
    #     else:
    #         return 0  

    def bfloat16_to_decimal(hex_str):
        # 32 비트 부동 소수점 값을 hex 문자열로 변환
        float32_hex = hex_str.ljust(8,'0')
        hex_data = bytes.fromhex(float32_hex)
        # hex 문자열을 부동 소수점 값으로 언팩
        decimal_value = struct.unpack('!f', hex_data)[0]

        return decimal_value
    
    decimal_array = np.vectorize(bfloat16_to_decimal)(convert_ar).reshape(Batch_size, origin_size//Batch_size)
    '''
    def to_decimal(num):
        # Perform your desired conversion logic here if needed
        return float(num)  # This is a basic example assuming conversion to integer

    # decimal_array = np.vectorize(to_decimal(final_ar))
    decimal_vectorized = np.vectorize(to_decimal)

    # Apply the vectorized function to your array
    decimal_array = decimal_vectorized(convert_ar).reshape(Batch_size, origin_size//Batch_size)

    outlist = outlist.tolist()
    decimal_array_sum = decimal_array[0] + decimal_array[1] + decimal_array[2] + decimal_array[3] + decimal_array[4] + decimal_array[5] + decimal_array[6] + decimal_array[7]

    # outlist = []
    outlist = decimal_array_sum.tolist()
    return outlist


def Weight_Gradient_Hardware_ReOrdering_whole_image_layer8(Out_CH, In_CH, DataCH0_List, DataCH1_List):

    # Filter_Num = 128
    # In_Channel_Num = 1024
    Filter_Num = Out_CH
    In_Channel_Num = In_CH
    Batch_size = 8

    origin0 = pd.DataFrame(DataCH0_List)
    origin1 = pd.DataFrame(DataCH1_List)

    origin_ar0 = np.array(origin0)
    origin_ar1 = np.array(origin1)
    half_size = np.size(origin_ar0)//(16)
    kenel_size = int(half_size/(Batch_size*Filter_Num*In_Channel_Num/16/2))

    # convert_ar0 = np.repeat('0000',A).reshape(Batch_size*half_size,16)
    # convert_ar1 = np.repeat('0000',A).reshape(Batch_size*half_size,16)
    # origin_ar = np.repeat('0000',A*2).reshape(half_size*2,16)
    # convert_ar = np.repeat('0000',A*2).reshape(half_size//(kenel_size*8),kenel_size*16,16)

    convert_ar0 = origin_ar0.reshape(half_size//(kenel_size*8),kenel_size*8*16)
    convert_ar1 = origin_ar1.reshape(half_size//(kenel_size*8),kenel_size*8*16)

    origin_ar = np.concatenate((convert_ar0, convert_ar1), axis=1).reshape(Batch_size,Filter_Num//16,In_Channel_Num//2,kenel_size*2,16)

    origin_size = Batch_size*Filter_Num*In_Channel_Num*kenel_size
    zero = '0000'

    Wconvert_array = np.repeat(zero, origin_size).reshape(Batch_size,Filter_Num//8,In_Channel_Num//4,kenel_size*2,16)

    for b in range (0,Batch_size):
        for k in range (0,int(Filter_Num/16)):
            for i in range (0,int(In_Channel_Num/4)):
                Wconvert_array[b][2*k][i]  = origin_ar[b][k][2*i]
                Wconvert_array[b][2*k+1][i]= origin_ar[b][k][2*i+1]



    Wconvert_array = Wconvert_array.reshape(Batch_size,Filter_Num//8,In_Channel_Num//4,2,kenel_size,16)



    filter_ar2 = np.repeat(zero, origin_size).reshape(Batch_size,Filter_Num//8,2,In_Channel_Num//4,kenel_size,16)
    filter_ar2 = Wconvert_array.transpose(0,1,3,2,4,5)

    filter_ar = np.repeat(zero, origin_size).reshape(Batch_size, Filter_Num//8, 2, 4, In_Channel_Num//4, kenel_size, 4)


    filter_ar[:, :, :, 0, :, :, :] = filter_ar2[:, :, :, :, :, 0:4]
    filter_ar[:, :, :, 1, :, :, :] = filter_ar2[:, :, :, :, :, 4:8]
    filter_ar[:, :, :, 2, :, :, :] = filter_ar2[:, :, :, :, :, 8:12]
    filter_ar[:, :, :, 3, :, :, :] = filter_ar2[:, :, :, :, :, 12:16]


    convert_ar = np.repeat(zero, origin_size).reshape(Batch_size, Filter_Num//8, 2, 4, In_Channel_Num//4, 4, kenel_size) # Using "2" make 8 outch



    # Transpose and assign the values directly
    convert_ar = filter_ar.transpose(0, 1, 2, 3, 4, 6, 5).reshape(Batch_size, origin_size//8)
    
  

    # def bfloat16_to_decimal(bfloat16_data):
    #     int_data = int(bfloat16_data, 16)
    #     sign = 1 if (int_data & 0x8000) == 0 else -1
    #     exponent = ((int_data & 0x7F80) >> 7) - 127
    #     fraction = 1.0 + (int_data & 0x007F) / 128.0
    #     if(int_data != 0):
    #         return sign * fraction * (2 ** exponent)
    #     else:
    #         return 0  

    def bfloat16_to_decimal(hex_str):
        # 32 비트 부동 소수점 값을 hex 문자열로 변환
        float32_hex = hex_str.ljust(8,'0')
        hex_data = bytes.fromhex(float32_hex)
        # hex 문자열을 부동 소수점 값으로 언팩
        decimal_value = struct.unpack('!f', hex_data)[0]

        return decimal_value

    # bfloat16_array = np.array(origin_ar, dtype=np.uint16)  # bfloat16 데이터
    decimal_array0 = np.vectorize(bfloat16_to_decimal)(convert_ar[0])
    decimal_array1 = np.vectorize(bfloat16_to_decimal)(convert_ar[1])
    decimal_array2 = np.vectorize(bfloat16_to_decimal)(convert_ar[2])
    decimal_array3 = np.vectorize(bfloat16_to_decimal)(convert_ar[3])
    decimal_array4 = np.vectorize(bfloat16_to_decimal)(convert_ar[4])
    decimal_array5 = np.vectorize(bfloat16_to_decimal)(convert_ar[5])
    decimal_array6 = np.vectorize(bfloat16_to_decimal)(convert_ar[6])
    decimal_array7 = np.vectorize(bfloat16_to_decimal)(convert_ar[7])

    # outlist = []
    outlist0 = decimal_array0.tolist()
    outlist1 = decimal_array1.tolist()
    outlist2 = decimal_array2.tolist()
    outlist3 = decimal_array3.tolist()
    outlist4 = decimal_array4.tolist()
    outlist5 = decimal_array5.tolist()
    outlist6 = decimal_array6.tolist()
    outlist7 = decimal_array7.tolist()
    return outlist0, outlist1, outlist2, outlist3, outlist4, outlist5, outlist6, outlist7


# Soft2Harward ReOrdering: Backward
# Weight Layer0 ReOrdering: Backward
def Weight_Hardware_Backward_ReOrdering_Layer0(Filter_Num, In_Channel_Num, Weight_List, Backward_List, Average_List):
    # Filter_Num_T = 128
    # In_Channel_Num_T = 1024
    Filter_Num_T = Filter_Num
    In_Channel_Num_T = In_Channel_Num

    Filter_Num = In_Channel_Num_T
    In_Channel_Num = Filter_Num_T

    A = int(Filter_Num/8)
    B = int(In_Channel_Num/4)
    C = int((Filter_Num*In_Channel_Num*9)/16)
    D = int(Filter_Num*In_Channel_Num*9)

    E = int(Filter_Num*4)
    F = int(Filter_Num/4)
    G = int(A*B*20*16)
    H = int(A*B*20)
    zero = '0000'
    # ------------------------------ All files require a garbage value in the first line. (ex) Add "test" to the first line---------------------------

    origin    =    pd.DataFrame(Weight_List)
    batch_add =    pd.DataFrame(Average_List)
    batch_mul =    pd.DataFrame(Backward_List)
    # bias_active =    pd.DataFrame(Active_List)

    bias_active_ar = np.repeat('3DCD',In_Channel_Num).reshape(In_Channel_Num,1)  # outchannel num == filter num

    # bias_active_ar = np.array(bias_active)
    batch_add_ar = np.array(batch_add)
    batch_mul_ar = np.array(batch_mul)
    origin_ar = np.array(origin)
    origin_ar = origin_ar.reshape(16,3,9)

    zero = '0000'
    zero_ar = np.repeat(zero,1872) # 16*13*9
    zero_ar = zero_ar.reshape(16,13,9)

    temp = np.repeat(zero,2304) # 16*16*9
    temp = temp.reshape(16,16,9)
    for i in range (0,16):
        temp[i] = np.concatenate( (origin_ar[i],zero_ar[i]), axis= 0)

    origin_ar = temp
    #origin_ar = np.array(origin_ar)

    origin_size = np.size(origin_ar)
    kenel_size = int(origin_size / (Filter_Num * In_Channel_Num))

    # origin_ar_T = np.repeat(zero,Filter_Num_T*In_Channel_Num_T*kenel_size).reshape(Filter_Num,In_Channel_Num,int(kenel_size**(1/2)),int(kenel_size**(1/2)))
    # origin_ar_TT = np.repeat(zero,Filter_Num_T*In_Channel_Num_T*kenel_size).reshape(Filter_Num,In_Channel_Num,int(kenel_size**(1/2)),int(kenel_size**(1/2)))
    # origin_ar = origin_ar.reshape(Filter_Num_T,In_Channel_Num_T,int(kenel_size**(1/2)),int(kenel_size**(1/2)))
    # origin_ar_T = origin_ar.transpose(1,0,2,3)
    # for i in range (0,Filter_Num):
    #     for j in range (0,In_Channel_Num):
    #         for k in range (0,int(kenel_size**(1/2))):
    #             for kk in range (0,int(kenel_size**(1/2))):
    #                 origin_ar_TT[i][j][int(kenel_size**(1/2))-1-k][int(kenel_size**(1/2))-1-kk] = origin_ar_T[i][j][k][kk]

    origin_ar_T = np.repeat(zero,Filter_Num_T*In_Channel_Num_T*kenel_size).reshape(Filter_Num,In_Channel_Num,kenel_size)
    origin_ar = origin_ar.reshape(Filter_Num_T,In_Channel_Num_T,kenel_size)
    origin_ar_T = origin_ar.transpose(1,0,2)
    origin_ar_TT = origin_ar_T[:,:,::-1]
    # origin_ar_TT = origin_ar_T

    # if (kenel_size==1) :
    #     origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,1) 
    #     zero = '0000'
    #     zero_ar = np.repeat(zero,6)
    #     zero_ar = zero_ar.reshape(6)
    #     temp_ar = np.repeat(zero, D).reshape(A,2,4,B,4,9)
    #     for i in range (0,A):
    #         for j in range (0,2):
    #             for k in range (0,4):
    #                 for l in range (0,B):
    #                     for m in range (0,4):
    #                         temp_ar[i][j][k][l][m] = np.concatenate( (zero_ar,origin_ar_TT[i][j][k][l][m],origin_ar_TT[i][j][k][l][m],origin_ar_TT[i][j][k][l][m]), axis = 0)
    #         # concat 4 in_channel
    #     filter_ar = np.repeat(zero, D).reshape(A,2,4,B,9,4)
    #     for i in range (0,A):
    #         for j in range (0,2):
    #             for k in range (0,4):
    #                 for l in range (0,B):
    #                     filter_ar[i][j][k][l] = temp_ar[i][j][k][l].T
    #         # print('aaaa')
    # elif (kenel_size==9) :
    #     origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,9) # Using "2" make 8 outch
    #     zero = '0000'
    #     #concat 4 in channel
    #     filter_ar = np.repeat(zero, origin_size).reshape(A,2,4,B,9,4)
    #     for fn in range (0,A):
    #         for fc2 in range (0,2):
    #             for fc1 in range (0,4):
    #                 for incn in range (0,B):
    #                     filter_ar[fn][fc2][fc1][incn] = origin_ar_TT[fn][fc2][fc1][incn].T
    # # print('ssss')
    if (kenel_size==1) :
        origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,1) #(filter/8 ,2(아래로 반복), 4(4개있음), input channel/4, 4(4개를 가져감), kernel)
        zero_ar = np.repeat(zero,A*2*4*B*4*6).reshape(A,2,4,B,4,6)
        temp_ar = np.repeat(zero, D).reshape(A,2,4,B,4,9)
        temp_ar = np.concatenate( (zero_ar,origin_ar_TT,origin_ar_TT,origin_ar_TT) , axis = 5)
        # concat 4 in_channel
        filter_ar = np.repeat(zero, D).reshape(A,2,4,B,9,4)
        filter_ar = temp_ar.transpose(0,1,2,3,5,4)
    elif (kenel_size==9) :
        origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,9) # Using "2" make 8 outch
        #concat 4 in channel
        filter_ar = np.repeat(zero, origin_size).reshape(A,2,4,B,9,4)
        filter_ar = origin_ar_TT.transpose(0,1,2,3,5,4)


    # to concat 4 filter
    # filter_ar2 = np.repeat(zero, D).reshape(A,2,B,9,16)
    # for fn in range(0,A):
    #     for fin in range (0,2):
    #         for cn in range (0,B):
    #             for d in range (0,9):
    #                 filter_ar2[fn][fin][cn][d] = np.concatenate( (filter_ar[fn][fin][0][cn][d],filter_ar[fn][fin][1][cn][d],filter_ar[fn][fin][2][cn][d],filter_ar[fn][fin][3][cn][d]), axis= 0)

    filter_ar2 = np.repeat(zero, D).reshape(A,2,B,9,16)

    filter_ar_0 = filter_ar[:,:,0,:,:,:]
    filter_ar_1 = filter_ar[:,:,1,:,:,:]
    filter_ar_2 = filter_ar[:,:,2,:,:,:]
    filter_ar_3 = filter_ar[:,:,3,:,:,:]

    filter_ar2 = np.concatenate((filter_ar_0, filter_ar_1, filter_ar_2, filter_ar_3), axis= 4)

    filter_ar3 = np.repeat(zero, D).reshape(A,B,2,9,16)  # to concat filter twice
    filter_ar3 = filter_ar2.transpose(0,2,1,3,4)

    # Wconvert_ar = filter_ar3.reshape(C,16)

    space = np.repeat(zero,1).reshape(1,1)
    vector_array = np.repeat(zero,B*16).reshape(int(B/2),2,16) #inchannel/8,2,16



    for i in range(0,int(B/2)): # filter_num / 8
        vector_array[i][0] = np.concatenate( (bias_active_ar[i*8],bias_active_ar[i*8+1],bias_active_ar[i*8+2],bias_active_ar[i*8+3],   batch_add_ar[i*8],batch_add_ar[i*8+1],batch_add_ar[i*8+2],batch_add_ar[i*8+3],
                                                bias_active_ar[i*8+4],bias_active_ar[i*8+5],bias_active_ar[i*8+6],bias_active_ar[i*8+7], batch_add_ar[i*8+4],batch_add_ar[i*8+5],batch_add_ar[i*8+6],batch_add_ar[i*8+7] ),axis=0)
        vector_array[i][1] = np.concatenate( (batch_mul_ar[i*8],batch_mul_ar[i*8+1],batch_mul_ar[i*8+2],batch_mul_ar[i*8+3],   space[0],space[0],space[0],space[0],
                                                batch_mul_ar[i*8+4],batch_mul_ar[i*8+5],batch_mul_ar[i*8+6],batch_mul_ar[i*8+7], space[0],space[0],space[0],space[0] ),axis=0)


    # vector_array = vector_array.reshape(B,16)
    zero = '0000'

    Wconvert_array = filter_ar3.reshape(A,B,18,16) #fil/8, in/4
    # biacsc_ar = np.repeat(zero,B*16).reshape(int(B/2),1,2,16)
    biacsc_ar = vector_array.reshape(int(B/2),1,2,16)
    concat_ar = np.repeat(zero,G).reshape(int(A/2),B*2,20,16)

    for k in range (0,int(A/2)):
        for i in range (0,B):
            concat_ar[k][2*i]   = np.concatenate((Wconvert_array[2*k][i],biacsc_ar[int(i/4)*2][0]), axis=0)
            concat_ar[k][2*i+1] = np.concatenate((Wconvert_array[2*k+1][i],biacsc_ar[int(i/4)*2+1][0]), axis=0)

    convert_ar = np.repeat(zero,G).reshape(H,16)
    convert_ar = concat_ar.reshape(H,16)


    Weight_List = []
    Weight_List.clear()
    for value in convert_ar:
        Result = ''.join(value)
        Weight_List.append(Result)

    df = pd.DataFrame(Weight_List)
    df1, df2 = Separated_Weight_DDR_Channel(df)
    #   df1.to_csv(Write_Path_Ch0, index=False, header=False, sep='\t')
    #   df2.to_csv(Write_Path_Ch1, index=False, header=False, sep='\t')
    Weight_Layer0_Channel0 = df1.values.tolist()
    Weight_Layer0_Channel1 = df2.values.tolist()
    Weight_Layer0 = [Weight_Layer0_Channel0, Weight_Layer0_Channel1]
    return Weight_Layer0

# Soft2Hardware ReOrdering: Backward
# Weight_OtherLayer ReOrdering: Backward
'''
def Weight_Hardware_Backward_ReOrdering_Layer8(Filter_Num, In_Channel_Num, Weight_List, Backward_List, Average_List):
    # Filter_Num_T = 128
    # In_Channel_Num_T = 1024
    Filter_Num_T = Filter_Num
    In_Channel_Num_T = In_Channel_Num

    Filter_Num = In_Channel_Num_T
    In_Channel_Num = Filter_Num_T

    A = int(Filter_Num/8)
    B = int(In_Channel_Num/4)
    C = int((Filter_Num*In_Channel_Num*9)/16)
    D = int(Filter_Num*In_Channel_Num*9)

    E = int(Filter_Num*4)
    F = int(Filter_Num/4)
    G = int(A*B*20*16)
    H = int(A*B*20)
    zero = '0000'

    origin = pd.DataFrame(Weight_List)
    #--------------------------------Activation and BatchNorm-----------------------
    batch_add =    pd.DataFrame(Average_List)
    batch_mul =    pd.DataFrame(Backward_List)

    bias_active_ar = np.repeat('0000',In_Channel_Num).reshape(In_Channel_Num,1)  # outchannel num == filter num

    batch_add_ar = np.array(batch_add)
    batch_mul_ar = np.array(batch_mul)

    origin_ar = np.array(origin)

    zero = '0000'
    origin_size = np.size(origin_ar)
    kenel_size = int(origin_size / (Filter_Num * In_Channel_Num))

    #   origin_ar_T = np.repeat(zero,Filter_Num_T*In_Channel_Num_T*kenel_size).reshape(Filter_Num,In_Channel_Num,int(kenel_size**(1/2)),int(kenel_size**(1/2)))
    #   origin_ar_TT = np.repeat(zero,Filter_Num_T*In_Channel_Num_T*kenel_size).reshape(Filter_Num,In_Channel_Num,int(kenel_size**(1/2)),int(kenel_size**(1/2)))
    #   origin_ar = origin_ar.reshape(Filter_Num_T,In_Channel_Num_T,int(kenel_size**(1/2)),int(kenel_size**(1/2)))
    #   origin_ar_T = origin_ar.transpose(1,0,2,3)
    #   for i in range (0,Filter_Num):
    #     for j in range (0,In_Channel_Num):
    #       for k in range (0,int(kenel_size**(1/2))):
    #         for kk in range (0,int(kenel_size**(1/2))):
    #           origin_ar_TT[i][j][int(kenel_size**(1/2))-1-k][int(kenel_size**(1/2))-1-kk] = origin_ar_T[i][j][k][kk]

    origin_ar_T = np.repeat(zero,Filter_Num_T*In_Channel_Num_T*kenel_size).reshape(Filter_Num,In_Channel_Num,kenel_size)
    origin_ar = origin_ar.reshape(Filter_Num_T,In_Channel_Num_T,kenel_size)
    origin_ar_T = origin_ar.transpose(1,0,2)
    origin_ar_TT = origin_ar_T[:,:,::-1]
    # origin_ar_TT = origin_ar_T

    # if (kenel_size==1) :
    #     origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,1) 
    #     zero = '0000'
    #     zero_ar = np.repeat(zero,6)
    #     zero_ar = zero_ar.reshape(6)
    #     temp_ar = np.repeat(zero, D).reshape(A,2,4,B,4,9)
    #     for i in range (0,A):
    #         for j in range (0,2):
    #             for k in range (0,4):
    #                 for l in range (0,B):
    #                     for m in range (0,4):
    #                         temp_ar[i][j][k][l][m] = np.concatenate( (zero_ar,origin_ar_TT[i][j][k][l][m],origin_ar_TT[i][j][k][l][m],origin_ar_TT[i][j][k][l][m]), axis = 0)
    #     # concat 4 in_channel
    #     filter_ar = np.repeat(zero, D).reshape(A,2,4,B,9,4)
    #     for i in range (0,A):
    #         for j in range (0,2):
    #             for k in range (0,4):
    #                 for l in range (0,B):
    #                     filter_ar[i][j][k][l] = temp_ar[i][j][k][l].T
    #     # print('aaaa')
    # elif (kenel_size==9) :
    #     origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,9) # Using "2" make 8 outch
    #     zero = '0000'
    #     #concat 4 in channel
    #     filter_ar = np.repeat(zero, origin_size).reshape(A,2,4,B,9,4)
    #     for fn in range (0,A):
    #         for fc2 in range (0,2):
    #             for fc1 in range (0,4):
    #                 for incn in range (0,B):
    #                     filter_ar[fn][fc2][fc1][incn] = origin_ar_TT[fn][fc2][fc1][incn].T
    # print('ssss')
    if (kenel_size==1) :
        origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,1) #(filter/8 ,2(아래로 반복), 4(4개있음), input channel/4, 4(4개를 가져감), kernel)
        zero_ar = np.repeat(zero,A*2*4*B*4*6).reshape(A,2,4,B,4,6)
        temp_ar = np.repeat(zero, D).reshape(A,2,4,B,4,9)
        temp_ar = np.concatenate( (zero_ar,origin_ar_TT,origin_ar_TT,origin_ar_TT) , axis = 5)
        # concat 4 in_channel
        filter_ar = np.repeat(zero, D).reshape(A,2,4,B,9,4)
        filter_ar = temp_ar.transpose(0,1,2,3,5,4)
    elif (kenel_size==9) :
        origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,9) # Using "2" make 8 outch
        #concat 4 in channel
        filter_ar = np.repeat(zero, origin_size).reshape(A,2,4,B,9,4)
        filter_ar = origin_ar_TT.transpose(0,1,2,3,5,4)


    # to concat 4 filter
    # filter_ar2 = np.repeat(zero, D).reshape(A,2,B,9,16)
    # for fn in range(0,A):
    #     for fin in range (0,2):
    #         for cn in range (0,B):
    #             for d in range (0,9):
    #                 filter_ar2[fn][fin][cn][d] = np.concatenate( (filter_ar[fn][fin][0][cn][d],filter_ar[fn][fin][1][cn][d],filter_ar[fn][fin][2][cn][d],filter_ar[fn][fin][3][cn][d]), axis= 0)

    filter_ar2 = np.repeat(zero, D).reshape(A,2,B,9,16)

    filter_ar_0 = filter_ar[:,:,0,:,:,:]
    filter_ar_1 = filter_ar[:,:,1,:,:,:]
    filter_ar_2 = filter_ar[:,:,2,:,:,:]
    filter_ar_3 = filter_ar[:,:,3,:,:,:]

    filter_ar2 = np.concatenate((filter_ar_0, filter_ar_1, filter_ar_2, filter_ar_3), axis= 4)

    filter_ar3 = np.repeat(zero, D).reshape(A,B,2,9,16)  # to concat filter twice
    filter_ar3 = filter_ar2.transpose(0,2,1,3,4)

    # Wconvert_ar = filter_ar3.reshape(C,16)

    space = np.repeat(zero,1).reshape(1,1)
    vector_array = np.repeat(zero,B*16).reshape(int(B/2),2,16) #inchannel/8,2,16


    for i in range(0,int(B/2)): # filter_num / 8
        vector_array[i][0] = np.concatenate( (bias_active_ar[i*8],bias_active_ar[i*8+1],bias_active_ar[i*8+2],bias_active_ar[i*8+3],   batch_add_ar[i*8],batch_add_ar[i*8+1],batch_add_ar[i*8+2],batch_add_ar[i*8+3],
                                                bias_active_ar[i*8+4],bias_active_ar[i*8+5],bias_active_ar[i*8+6],bias_active_ar[i*8+7], batch_add_ar[i*8+4],batch_add_ar[i*8+5],batch_add_ar[i*8+6],batch_add_ar[i*8+7] ),axis=0)
        vector_array[i][1] = np.concatenate( (batch_mul_ar[i*8],batch_mul_ar[i*8+1],batch_mul_ar[i*8+2],batch_mul_ar[i*8+3],   space[0],space[0],space[0],space[0],
                                            batch_mul_ar[i*8+4],batch_mul_ar[i*8+5],batch_mul_ar[i*8+6],batch_mul_ar[i*8+7], space[0],space[0],space[0],space[0] ),axis=0)


    # vector_array = vector_array.reshape(B,16)
    # zero = '0000'

    Wconvert_array = filter_ar3.reshape(A,B,18,16) #fil/8, in/4
    # biacsc_ar = np.repeat(zero,B*16).reshape(int(B/2),1,2,16)
    biacsc_ar = vector_array.reshape(int(B/2),1,2,16)
    concat_ar = np.repeat(zero,G).reshape(int(A/2),B*2,20,16)

    for k in range (0,int(A/2)):
        for i in range (0,B):
            concat_ar[k][2*i]   = np.concatenate((Wconvert_array[2*k][i],biacsc_ar[int(i/4)*2][0]), axis=0)
            concat_ar[k][2*i+1] = np.concatenate((Wconvert_array[2*k+1][i],biacsc_ar[int(i/4)*2+1][0]), axis=0)

    convert_ar = np.repeat(zero,G).reshape(H,16)
    convert_ar = concat_ar.reshape(H,16)


    Weight_List = []
    Weight_List.clear()
    for value in convert_ar:
        Result = ''.join(value)
        Weight_List.append(Result)

    df = pd.DataFrame(Weight_List)
    df1, df2 = Separated_Weight_DDR_Channel(df)
    #   df1.to_csv(Write_Path_Ch0, index=False, header=False, sep='\t')
    #   df2.to_csv(Write_Path_Ch1, index=False, header=False, sep='\t')
    Weight_OtherLayer_Channel0 = df1.values.tolist()
    Weight_OtherLayer_Channel1 = df2.values.tolist()
    Weight_OtherLayer = [Weight_OtherLayer_Channel0, Weight_OtherLayer_Channel1]
    return Weight_OtherLayer


def Weight_Hardware_Backward_ReOrdering_OtherLayer(Filter_Num, In_Channel_Num, Weight_List, Backward_List, Average_List, Active_List):
    # Filter_Num_T = 128
    # In_Channel_Num_T = 1024
    Filter_Num_T = Filter_Num
    In_Channel_Num_T = In_Channel_Num

    Filter_Num = In_Channel_Num_T
    In_Channel_Num = Filter_Num_T

    A = int(Filter_Num/8)
    B = int(In_Channel_Num/4)
    C = int((Filter_Num*In_Channel_Num*9)/16)
    D = int(Filter_Num*In_Channel_Num*9)

    E = int(Filter_Num*4)
    F = int(Filter_Num/4)
    G = int(A*B*20*16)
    H = int(A*B*20)
    zero = '0000'

    origin = pd.DataFrame(Weight_List)
    #--------------------------------Activation and BatchNorm-----------------------
    bias_active =  pd.DataFrame(Active_List)
    batch_add =    pd.DataFrame(Average_List)
    batch_mul =    pd.DataFrame(Backward_List)

    # bias_active_ar = np.repeat('3DCD',In_Channel_Num).reshape(In_Channel_Num,1)  # outchannel num == filter num

    bias_active_ar = np.array(bias_active)
    batch_add_ar = np.array(batch_add)
    batch_mul_ar = np.array(batch_mul)

    origin_ar = np.array(origin)

    zero = '0000'
    origin_size = np.size(origin_ar)
    kenel_size = int(origin_size / (Filter_Num * In_Channel_Num))

    #   origin_ar_T = np.repeat(zero,Filter_Num_T*In_Channel_Num_T*kenel_size).reshape(Filter_Num,In_Channel_Num,int(kenel_size**(1/2)),int(kenel_size**(1/2)))
    #   origin_ar_TT = np.repeat(zero,Filter_Num_T*In_Channel_Num_T*kenel_size).reshape(Filter_Num,In_Channel_Num,int(kenel_size**(1/2)),int(kenel_size**(1/2)))
    #   origin_ar = origin_ar.reshape(Filter_Num_T,In_Channel_Num_T,int(kenel_size**(1/2)),int(kenel_size**(1/2)))
    #   origin_ar_T = origin_ar.transpose(1,0,2,3)
    #   for i in range (0,Filter_Num):
    #     for j in range (0,In_Channel_Num):
    #       for k in range (0,int(kenel_size**(1/2))):
    #         for kk in range (0,int(kenel_size**(1/2))):
    #           origin_ar_TT[i][j][int(kenel_size**(1/2))-1-k][int(kenel_size**(1/2))-1-kk] = origin_ar_T[i][j][k][kk]

    origin_ar_T = np.repeat(zero,Filter_Num_T*In_Channel_Num_T*kenel_size).reshape(Filter_Num,In_Channel_Num,kenel_size)
    origin_ar = origin_ar.reshape(Filter_Num_T,In_Channel_Num_T,kenel_size)
    origin_ar_T = origin_ar.transpose(1,0,2)
    origin_ar_TT = origin_ar_T[:,:,::-1]
    # origin_ar_TT = origin_ar_T

    # if (kenel_size==1) :
    #     origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,1) 
    #     zero = '0000'
    #     zero_ar = np.repeat(zero,6)
    #     zero_ar = zero_ar.reshape(6)
    #     temp_ar = np.repeat(zero, D).reshape(A,2,4,B,4,9)
    #     for i in range (0,A):
    #         for j in range (0,2):
    #             for k in range (0,4):
    #                 for l in range (0,B):
    #                     for m in range (0,4):
    #                         temp_ar[i][j][k][l][m] = np.concatenate( (zero_ar,origin_ar_TT[i][j][k][l][m],origin_ar_TT[i][j][k][l][m],origin_ar_TT[i][j][k][l][m]), axis = 0)
    #     # concat 4 in_channel
    #     filter_ar = np.repeat(zero, D).reshape(A,2,4,B,9,4)
    #     for i in range (0,A):
    #         for j in range (0,2):
    #             for k in range (0,4):
    #                 for l in range (0,B):
    #                     filter_ar[i][j][k][l] = temp_ar[i][j][k][l].T
    #     # print('aaaa')
    # elif (kenel_size==9) :
    #     origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,9) # Using "2" make 8 outch
    #     zero = '0000'
    #     #concat 4 in channel
    #     filter_ar = np.repeat(zero, origin_size).reshape(A,2,4,B,9,4)
    #     for fn in range (0,A):
    #         for fc2 in range (0,2):
    #             for fc1 in range (0,4):
    #                 for incn in range (0,B):
    #                     filter_ar[fn][fc2][fc1][incn] = origin_ar_TT[fn][fc2][fc1][incn].T
    # print('ssss')
    if (kenel_size==1) :
        origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,1) #(filter/8 ,2(아래로 반복), 4(4개있음), input channel/4, 4(4개를 가져감), kernel)
        zero_ar = np.repeat(zero,A*2*4*B*4*6).reshape(A,2,4,B,4,6)
        temp_ar = np.repeat(zero, D).reshape(A,2,4,B,4,9)
        temp_ar = np.concatenate( (zero_ar,origin_ar_TT,origin_ar_TT,origin_ar_TT) , axis = 5)
        # concat 4 in_channel
        filter_ar = np.repeat(zero, D).reshape(A,2,4,B,9,4)
        filter_ar = temp_ar.transpose(0,1,2,3,5,4)
    elif (kenel_size==9) :
        origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,9) # Using "2" make 8 outch
        #concat 4 in channel
        filter_ar = np.repeat(zero, origin_size).reshape(A,2,4,B,9,4)
        filter_ar = origin_ar_TT.transpose(0,1,2,3,5,4)


    # to concat 4 filter
    # filter_ar2 = np.repeat(zero, D).reshape(A,2,B,9,16)
    # for fn in range(0,A):
    #     for fin in range (0,2):
    #         for cn in range (0,B):
    #             for d in range (0,9):
    #                 filter_ar2[fn][fin][cn][d] = np.concatenate( (filter_ar[fn][fin][0][cn][d],filter_ar[fn][fin][1][cn][d],filter_ar[fn][fin][2][cn][d],filter_ar[fn][fin][3][cn][d]), axis= 0)

    filter_ar2 = np.repeat(zero, D).reshape(A,2,B,9,16)

    filter_ar_0 = filter_ar[:,:,0,:,:,:]
    filter_ar_1 = filter_ar[:,:,1,:,:,:]
    filter_ar_2 = filter_ar[:,:,2,:,:,:]
    filter_ar_3 = filter_ar[:,:,3,:,:,:]

    filter_ar2 = np.concatenate((filter_ar_0, filter_ar_1, filter_ar_2, filter_ar_3), axis= 4)

    filter_ar3 = np.repeat(zero, D).reshape(A,B,2,9,16)  # to concat filter twice
    filter_ar3 = filter_ar2.transpose(0,2,1,3,4)

    # Wconvert_ar = filter_ar3.reshape(C,16)

    space = np.repeat(zero,1).reshape(1,1)
    vector_array = np.repeat(zero,B*16).reshape(int(B/2),2,16) #inchannel/8,2,16


    for i in range(0,int(B/2)): # filter_num / 8
        vector_array[i][0] = np.concatenate( (bias_active_ar[i*8],bias_active_ar[i*8+1],bias_active_ar[i*8+2],bias_active_ar[i*8+3],   batch_add_ar[i*8],batch_add_ar[i*8+1],batch_add_ar[i*8+2],batch_add_ar[i*8+3],
                                                bias_active_ar[i*8+4],bias_active_ar[i*8+5],bias_active_ar[i*8+6],bias_active_ar[i*8+7], batch_add_ar[i*8+4],batch_add_ar[i*8+5],batch_add_ar[i*8+6],batch_add_ar[i*8+7] ),axis=0)
        vector_array[i][1] = np.concatenate( (batch_mul_ar[i*8],batch_mul_ar[i*8+1],batch_mul_ar[i*8+2],batch_mul_ar[i*8+3],   space[0],space[0],space[0],space[0],
                                            batch_mul_ar[i*8+4],batch_mul_ar[i*8+5],batch_mul_ar[i*8+6],batch_mul_ar[i*8+7], space[0],space[0],space[0],space[0] ),axis=0)


    # vector_array = vector_array.reshape(B,16)
    # zero = '0000'

    Wconvert_array = filter_ar3.reshape(A,B,18,16) #fil/8, in/4
    # biacsc_ar = np.repeat(zero,B*16).reshape(int(B/2),1,2,16)
    biacsc_ar = vector_array.reshape(int(B/2),1,2,16)
    concat_ar = np.repeat(zero,G).reshape(int(A/2),B*2,20,16)

    for k in range (0,int(A/2)):
        for i in range (0,B):
            concat_ar[k][2*i]   = np.concatenate((Wconvert_array[2*k][i],biacsc_ar[int(i/4)*2][0]), axis=0)
            concat_ar[k][2*i+1] = np.concatenate((Wconvert_array[2*k+1][i],biacsc_ar[int(i/4)*2+1][0]), axis=0)

    convert_ar = np.repeat(zero,G).reshape(H,16)
    convert_ar = concat_ar.reshape(H,16)


    Weight_List = []
    Weight_List.clear()
    for value in convert_ar:
        Result = ''.join(value)
        Weight_List.append(Result)

    df = pd.DataFrame(Weight_List)
    df1, df2 = Separated_Weight_DDR_Channel(df)
    #   df1.to_csv(Write_Path_Ch0, index=False, header=False, sep='\t')
    #   df2.to_csv(Write_Path_Ch1, index=False, header=False, sep='\t')
    Weight_OtherLayer_Channel0 = df1.values.tolist()
    Weight_OtherLayer_Channel1 = df2.values.tolist()
    Weight_OtherLayer = [Weight_OtherLayer_Channel0, Weight_OtherLayer_Channel1]
    return Weight_OtherLayer


# def Fmap_Ordering(Channel, Data_List):
    origin = pd.DataFrame(Data_List)
    Output_Channel = Channel
    origin_ar = np.array(origin)
    origin_size = np.size(origin_ar)
    Fmap_size = int((origin_size/Output_Channel)**(1/2))

    zero = "0000"

    # original fmap shape
    origin_ar = origin_ar.reshape(Output_Channel,Fmap_size,Fmap_size)

    # change row and col each other(전치)
    for i in range (0,Output_Channel):
        origin_ar[i] = origin_ar[i].T

    origin_ar = origin_ar.reshape(Output_Channel,Fmap_size,Fmap_size)
    iter_13 = int(Fmap_size/13)

    concat_ar = np.repeat(zero, Output_Channel*Fmap_size*(Fmap_size+iter_13*3)).reshape(Output_Channel,(Fmap_size+iter_13*3),Fmap_size)

    for i in range (0,Output_Channel):
        for j in range (0,iter_13):
            for k in range (0,Fmap_size):
                concat_ar[i][j*16+0][k]  = origin_ar[i][j*13+0][k]
                concat_ar[i][j*16+1][k]  = origin_ar[i][j*13+1][k]
                concat_ar[i][j*16+2][k]  = origin_ar[i][j*13+2][k]
                concat_ar[i][j*16+3][k]  = origin_ar[i][j*13+3][k]
                concat_ar[i][j*16+4][k]  = origin_ar[i][j*13+4][k]
                concat_ar[i][j*16+5][k]  = origin_ar[i][j*13+5][k]
                concat_ar[i][j*16+6][k]  = origin_ar[i][j*13+6][k]
                concat_ar[i][j*16+7][k]  = origin_ar[i][j*13+7][k]
                concat_ar[i][j*16+8][k]  = origin_ar[i][j*13+8][k]
                concat_ar[i][j*16+9][k]  = origin_ar[i][j*13+9][k]
                concat_ar[i][j*16+10][k] = origin_ar[i][j*13+10][k]
                concat_ar[i][j*16+11][k] = origin_ar[i][j*13+11][k]
                concat_ar[i][j*16+12][k] = origin_ar[i][j*13+12][k]
                concat_ar[i][j*16+13][k] = origin_ar[i][j*13+12][k]
                concat_ar[i][j*16+14][k] = origin_ar[i][j*13+12][k]
                concat_ar[i][j*16+15][k] = origin_ar[i][j*13+12][k]
    concat_ar = concat_ar.reshape(Output_Channel*(Fmap_size+iter_13*3),Fmap_size,1)


    four_ar1 = np.repeat(zero,int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/4)).reshape(int(Output_Channel*(Fmap_size+iter_13*3)/4),Fmap_size,1)
    four_ar2 = np.repeat(zero,int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/4)).reshape(int(Output_Channel*(Fmap_size+iter_13*3)/4),Fmap_size,1)
    four_ar3 = np.repeat(zero,int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/4)).reshape(int(Output_Channel*(Fmap_size+iter_13*3)/4),Fmap_size,1)
    four_ar4 = np.repeat(zero,int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/4)).reshape(int(Output_Channel*(Fmap_size+iter_13*3)/4),Fmap_size,1)

    for i in range (0,int(Output_Channel*(Fmap_size+iter_13*3)/4)):
        for j in range (0,Fmap_size):
            four_ar1[i][j][0] = concat_ar[4*i][j][0]
            four_ar2[i][j][0] = concat_ar[4*i+1][j][0]
            four_ar3[i][j][0] = concat_ar[4*i+2][j][0]
            four_ar4[i][j][0] = concat_ar[4*i+3][j][0]

    four_ar1 = four_ar1.reshape(Output_Channel,int(Fmap_size*(Fmap_size+iter_13*3)/4),1)
    four_ar2 = four_ar2.reshape(Output_Channel,int(Fmap_size*(Fmap_size+iter_13*3)/4),1)
    four_ar3 = four_ar3.reshape(Output_Channel,int(Fmap_size*(Fmap_size+iter_13*3)/4),1)
    four_ar4 = four_ar4.reshape(Output_Channel,int(Fmap_size*(Fmap_size+iter_13*3)/4),1)

    fmap1 = np.repeat(zero,int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/4)).reshape(int(Output_Channel/16),int(Fmap_size*(Fmap_size+iter_13*3)/4),16)
    fmap2 = np.repeat(zero,int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/4)).reshape(int(Output_Channel/16),int(Fmap_size*(Fmap_size+iter_13*3)/4),16)
    fmap3 = np.repeat(zero,int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/4)).reshape(int(Output_Channel/16),int(Fmap_size*(Fmap_size+iter_13*3)/4),16)
    fmap4 = np.repeat(zero,int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/4)).reshape(int(Output_Channel/16),int(Fmap_size*(Fmap_size+iter_13*3)/4),16)

    # concat 4 input channel
    for i in range (0, int(Output_Channel/16)):
        fmap1[i] = np.concatenate( ( four_ar1[16*i+0], four_ar1[16*i+1], four_ar1[16*i+2], four_ar1[16*i+3], four_ar2[16*i+0], four_ar2[16*i+1], four_ar2[16*i+2], four_ar2[16*i+3], four_ar3[16*i+0], four_ar3[16*i+1], four_ar3[16*i+2], four_ar3[16*i+3], four_ar4[16*i+0], four_ar4[16*i+1], four_ar4[16*i+2], four_ar4[16*i+3]), axis=1)
        fmap2[i] = np.concatenate( ( four_ar1[16*i+4], four_ar1[16*i+5], four_ar1[16*i+6], four_ar1[16*i+7], four_ar2[16*i+4], four_ar2[16*i+5], four_ar2[16*i+6], four_ar2[16*i+7], four_ar3[16*i+4], four_ar3[16*i+5], four_ar3[16*i+6], four_ar3[16*i+7], four_ar4[16*i+4], four_ar4[16*i+5], four_ar4[16*i+6], four_ar4[16*i+7]), axis=1)
        fmap3[i] = np.concatenate( ( four_ar1[16*i+8], four_ar1[16*i+9], four_ar1[16*i+10], four_ar1[16*i+11], four_ar2[16*i+8], four_ar2[16*i+9], four_ar2[16*i+10], four_ar2[16*i+11], four_ar3[16*i+8], four_ar3[16*i+9], four_ar3[16*i+10], four_ar3[16*i+11], four_ar4[16*i+8], four_ar4[16*i+9], four_ar4[16*i+10], four_ar4[16*i+11]), axis=1)
        fmap4[i] = np.concatenate( ( four_ar1[16*i+12], four_ar1[16*i+13], four_ar1[16*i+14], four_ar1[16*i+15], four_ar2[16*i+12], four_ar2[16*i+13], four_ar2[16*i+14], four_ar2[16*i+15], four_ar3[16*i+12], four_ar3[16*i+13], four_ar3[16*i+14], four_ar3[16*i+15], four_ar4[16*i+12], four_ar4[16*i+13], four_ar4[16*i+14], four_ar4[16*i+15]), axis=1)

    # print(origin_size)
    # print(fmap1.shape)

    # # concat 4 pixel
    # convert_ar = np.repeat(zero,origin_size).reshape(int(origin_size/16),16)
    # for i in range (0,int(Output_Channel/16)) :
    #   for ch in range (0,16): # 16 고정
    #     for pix in range (0,int(Fmap_size*Fmap_size/4)):
    #       convert_ar [i*Fmap_size*Fmap_size+pix*4][ch]= fmap1.reshape(int(Output_Channel/4),int(Fmap_size*Fmap_size/4),16)[i*4+0][pix][ch]
    #       convert_ar [i*Fmap_size*Fmap_size+pix*4+1][ch]= fmap2.reshape(int(Output_Channel/4),int(Fmap_size*Fmap_size/4),16)[i*4+1][pix][ch]
    #       convert_ar [i*Fmap_size*Fmap_size+pix*4+2][ch]= fmap3.reshape(int(Output_Channel/4),int(Fmap_size*Fmap_size/4),16)[i*4+2][pix][ch]
    #       convert_ar [i*Fmap_size*Fmap_size+pix*4+3][ch]= fmap4.reshape(int(Output_Channel/4),int(Fmap_size*Fmap_size/4),16)[i*4+3][pix][ch]

    trans_ar = np.repeat(zero,Output_Channel*Fmap_size*(Fmap_size+iter_13*3)).reshape(int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/16),16)
    for i in range (0,int(Output_Channel/16)):
        for ch in range (0,16):
            for pix in range (0,int(Fmap_size*(Fmap_size+iter_13*3)/4)):
                trans_ar [i*Fmap_size*(Fmap_size+iter_13*3)+pix*4][ch]= fmap1[i][pix][ch]
                trans_ar [i*Fmap_size*(Fmap_size+iter_13*3)+pix*4+1][ch]= fmap2[i][pix][ch]
                trans_ar [i*Fmap_size*(Fmap_size+iter_13*3)+pix*4+2][ch]= fmap3[i][pix][ch]
                trans_ar [i*Fmap_size*(Fmap_size+iter_13*3)+pix*4+3][ch]= fmap4[i][pix][ch]

    Fmap_List = []
    Fmap_List.clear()
    for value in trans_ar:
        Result = ''.join(value)
        Fmap_List.append(Result)

    df = pd.DataFrame(Fmap_List)
    df1, df2 = Separated_Fmap_DDR_Channel(df)
#   df1.to_csv(Write_Path_Ch0, index=False, header=False, sep='\t')
#   df2.to_csv(Write_Path_Ch1, index=False, header=False, sep='\t')
    Image_OtherLayer_Channel0 = df1.values.tolist()
    Image_OtherLayer_Channel1 = df2.values.tolist()
    Image_OtherLayer = [Image_OtherLayer_Channel0, Image_OtherLayer_Channel1]
    return Image_OtherLayer
'''


def Weight_Hardware_Backward_ReOrdering_Layer8(Filter_Num, In_Channel_Num, Weight_List, Backward_List, Average_List):
    # Filter_Num_T = 128
    # In_Channel_Num_T = 1024
    Filter_Num_T = Filter_Num
    In_Channel_Num_T = In_Channel_Num

    Filter_Num = In_Channel_Num_T
    In_Channel_Num = Filter_Num_T

    A = int(Filter_Num/8)
    B = int(In_Channel_Num/4)
    C = int((Filter_Num*In_Channel_Num*9)/16)
    D = int(Filter_Num*In_Channel_Num*9)

    E = int(Filter_Num*4)
    F = int(Filter_Num/4)
    G = int(A*B*20*16)
    H = int(A*B*20)
    zero = '0000'

    origin = pd.DataFrame(Weight_List)
    #--------------------------------Activation and BatchNorm-----------------------
    batch_add =    pd.DataFrame(Average_List)
    batch_mul =    pd.DataFrame(Backward_List)

    bias_active_ar = np.repeat('0000',In_Channel_Num).reshape(In_Channel_Num,1)  # outchannel num == filter num

    batch_add_ar = np.array(batch_add)
    batch_mul_ar = np.array(batch_mul)

    origin_ar = np.array(origin)

    zero = '0000'
    origin_size = np.size(origin_ar)
    kenel_size = int(origin_size / (Filter_Num * In_Channel_Num))

    #   origin_ar_T = np.repeat(zero,Filter_Num_T*In_Channel_Num_T*kenel_size).reshape(Filter_Num,In_Channel_Num,int(kenel_size**(1/2)),int(kenel_size**(1/2)))
    #   origin_ar_TT = np.repeat(zero,Filter_Num_T*In_Channel_Num_T*kenel_size).reshape(Filter_Num,In_Channel_Num,int(kenel_size**(1/2)),int(kenel_size**(1/2)))
    #   origin_ar = origin_ar.reshape(Filter_Num_T,In_Channel_Num_T,int(kenel_size**(1/2)),int(kenel_size**(1/2)))
    #   origin_ar_T = origin_ar.transpose(1,0,2,3)
    #   for i in range (0,Filter_Num):
    #     for j in range (0,In_Channel_Num):
    #       for k in range (0,int(kenel_size**(1/2))):
    #         for kk in range (0,int(kenel_size**(1/2))):
    #           origin_ar_TT[i][j][int(kenel_size**(1/2))-1-k][int(kenel_size**(1/2))-1-kk] = origin_ar_T[i][j][k][kk]

    origin_ar_T = np.repeat(zero,Filter_Num_T*In_Channel_Num_T*kenel_size).reshape(Filter_Num,In_Channel_Num,kenel_size)
    origin_ar = origin_ar.reshape(Filter_Num_T,In_Channel_Num_T,kenel_size)
    origin_ar_T = origin_ar.transpose(1,0,2)
    origin_ar_TT = origin_ar_T[:,:,::-1]
    # origin_ar_TT = origin_ar_T

    # if (kenel_size==1) :
    #     origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,1) 
    #     zero = '0000'
    #     zero_ar = np.repeat(zero,6)
    #     zero_ar = zero_ar.reshape(6)
    #     temp_ar = np.repeat(zero, D).reshape(A,2,4,B,4,9)
    #     for i in range (0,A):
    #         for j in range (0,2):
    #             for k in range (0,4):
    #                 for l in range (0,B):
    #                     for m in range (0,4):
    #                         temp_ar[i][j][k][l][m] = np.concatenate( (zero_ar,origin_ar_TT[i][j][k][l][m],origin_ar_TT[i][j][k][l][m],origin_ar_TT[i][j][k][l][m]), axis = 0)
    #     # concat 4 in_channel
    #     filter_ar = np.repeat(zero, D).reshape(A,2,4,B,9,4)
    #     for i in range (0,A):
    #         for j in range (0,2):
    #             for k in range (0,4):
    #                 for l in range (0,B):
    #                     filter_ar[i][j][k][l] = temp_ar[i][j][k][l].T
    #     # print('aaaa')
    # elif (kenel_size==9) :
    #     origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,9) # Using "2" make 8 outch
    #     zero = '0000'
    #     #concat 4 in channel
    #     filter_ar = np.repeat(zero, origin_size).reshape(A,2,4,B,9,4)
    #     for fn in range (0,A):
    #         for fc2 in range (0,2):
    #             for fc1 in range (0,4):
    #                 for incn in range (0,B):
    #                     filter_ar[fn][fc2][fc1][incn] = origin_ar_TT[fn][fc2][fc1][incn].T
    # print('ssss')
    if (kenel_size==1) :
        origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,1) #(filter/8 ,2(아래로 반복), 4(4개있음), input channel/4, 4(4개를 가져감), kernel)
        zero_ar = np.repeat(zero,A*2*4*B*4*6).reshape(A,2,4,B,4,6)
        temp_ar = np.repeat(zero, D).reshape(A,2,4,B,4,9)
        temp_ar = np.concatenate( (zero_ar,origin_ar_TT,origin_ar_TT,origin_ar_TT) , axis = 5)
        # concat 4 in_channel
        filter_ar = np.repeat(zero, D).reshape(A,2,4,B,9,4)
        filter_ar = temp_ar.transpose(0,1,2,3,5,4)
    elif (kenel_size==9) :
        origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,9) # Using "2" make 8 outch
        #concat 4 in channel
        filter_ar = np.repeat(zero, origin_size).reshape(A,2,4,B,9,4)
        filter_ar = origin_ar_TT.transpose(0,1,2,3,5,4)


    # to concat 4 filter
    # filter_ar2 = np.repeat(zero, D).reshape(A,2,B,9,16)
    # for fn in range(0,A):
    #     for fin in range (0,2):
    #         for cn in range (0,B):
    #             for d in range (0,9):
    #                 filter_ar2[fn][fin][cn][d] = np.concatenate( (filter_ar[fn][fin][0][cn][d],filter_ar[fn][fin][1][cn][d],filter_ar[fn][fin][2][cn][d],filter_ar[fn][fin][3][cn][d]), axis= 0)

    filter_ar2 = np.repeat(zero, D).reshape(A,2,B,9,16)

    filter_ar_0 = filter_ar[:,:,0,:,:,:]
    filter_ar_1 = filter_ar[:,:,1,:,:,:]
    filter_ar_2 = filter_ar[:,:,2,:,:,:]
    filter_ar_3 = filter_ar[:,:,3,:,:,:]

    filter_ar2 = np.concatenate((filter_ar_0, filter_ar_1, filter_ar_2, filter_ar_3), axis= 4)

    filter_ar3 = np.repeat(zero, D).reshape(A,B,2,9,16)  # to concat filter twice
    filter_ar3 = filter_ar2.transpose(0,2,1,3,4)

    # Wconvert_ar = filter_ar3.reshape(C,16)

    space = np.repeat(zero,1).reshape(1,1)
    vector_array = np.repeat(zero,B*16).reshape(int(B/2),2,16) #inchannel/8,2,16


    for i in range(0,int(B/2)): # filter_num / 8
        vector_array[i][0] = np.concatenate( (bias_active_ar[i*8],bias_active_ar[i*8+1],bias_active_ar[i*8+2],bias_active_ar[i*8+3],   batch_add_ar[i*8],batch_add_ar[i*8+1],batch_add_ar[i*8+2],batch_add_ar[i*8+3],
                                                bias_active_ar[i*8+4],bias_active_ar[i*8+5],bias_active_ar[i*8+6],bias_active_ar[i*8+7], batch_add_ar[i*8+4],batch_add_ar[i*8+5],batch_add_ar[i*8+6],batch_add_ar[i*8+7] ),axis=0)
        vector_array[i][1] = np.concatenate( (batch_mul_ar[i*8],batch_mul_ar[i*8+1],batch_mul_ar[i*8+2],batch_mul_ar[i*8+3],   space[0],space[0],space[0],space[0],
                                            batch_mul_ar[i*8+4],batch_mul_ar[i*8+5],batch_mul_ar[i*8+6],batch_mul_ar[i*8+7], space[0],space[0],space[0],space[0] ),axis=0)


    # vector_array = vector_array.reshape(B,16)
    # zero = '0000'

    Wconvert_array = filter_ar3.reshape(A,B,18,16) #fil/8, in/4
    # biacsc_ar = np.repeat(zero,B*16).reshape(int(B/2),1,2,16)
    biacsc_ar = vector_array.reshape(int(B/2),1,2,16)
    concat_ar = np.repeat(zero,G).reshape(int(A/2),B*2,20,16)

    for k in range (0,int(A/2)):
        for i in range (0,B):
            concat_ar[k][2*i]   = np.concatenate((Wconvert_array[2*k][i],biacsc_ar[int(i/4)*2][0]), axis=0)
            concat_ar[k][2*i+1] = np.concatenate((Wconvert_array[2*k+1][i],biacsc_ar[int(i/4)*2+1][0]), axis=0)

    convert_ar = np.repeat(zero,G).reshape(H,16)
    convert_ar = concat_ar.reshape(H,16)


    Weight_List = []
    Weight_List.clear()
    for value in convert_ar:
        Result = ''.join(value)
        Weight_List.append(Result)

    df = pd.DataFrame(Weight_List)
    df1, df2 = Separated_Weight_DDR_Channel(df)
    #   df1.to_csv(Write_Path_Ch0, index=False, header=False, sep='\t')
    #   df2.to_csv(Write_Path_Ch1, index=False, header=False, sep='\t')
    Weight_OtherLayer_Channel0 = df1.values.tolist()
    Weight_OtherLayer_Channel1 = df2.values.tolist()
    Weight_OtherLayer = [Weight_OtherLayer_Channel0, Weight_OtherLayer_Channel1]
    return Weight_OtherLayer


def Weight_Hardware_Backward_ReOrdering_OtherLayer(Filter_Num, In_Channel_Num, Weight_List, Backward_List, Average_List):
    # Filter_Num_T = 128
    # In_Channel_Num_T = 1024
    Filter_Num_T = Filter_Num
    In_Channel_Num_T = In_Channel_Num

    Filter_Num = In_Channel_Num_T
    In_Channel_Num = Filter_Num_T

    A = int(Filter_Num/8)
    B = int(In_Channel_Num/4)
    C = int((Filter_Num*In_Channel_Num*9)/16)
    D = int(Filter_Num*In_Channel_Num*9)

    E = int(Filter_Num*4)
    F = int(Filter_Num/4)
    G = int(A*B*20*16)
    H = int(A*B*20)
    zero = '0000'

    origin = pd.DataFrame(Weight_List)
    #--------------------------------Activation and BatchNorm-----------------------
    batch_add =    pd.DataFrame(Average_List)
    batch_mul =    pd.DataFrame(Backward_List)

    bias_active_ar = np.repeat('3DCD',In_Channel_Num).reshape(In_Channel_Num,1)  # outchannel num == filter num

    batch_add_ar = np.array(batch_add)
    batch_mul_ar = np.array(batch_mul)

    origin_ar = np.array(origin)

    zero = '0000'
    origin_size = np.size(origin_ar)
    kenel_size = int(origin_size / (Filter_Num * In_Channel_Num))

    #   origin_ar_T = np.repeat(zero,Filter_Num_T*In_Channel_Num_T*kenel_size).reshape(Filter_Num,In_Channel_Num,int(kenel_size**(1/2)),int(kenel_size**(1/2)))
    #   origin_ar_TT = np.repeat(zero,Filter_Num_T*In_Channel_Num_T*kenel_size).reshape(Filter_Num,In_Channel_Num,int(kenel_size**(1/2)),int(kenel_size**(1/2)))
    #   origin_ar = origin_ar.reshape(Filter_Num_T,In_Channel_Num_T,int(kenel_size**(1/2)),int(kenel_size**(1/2)))
    #   origin_ar_T = origin_ar.transpose(1,0,2,3)
    #   for i in range (0,Filter_Num):
    #     for j in range (0,In_Channel_Num):
    #       for k in range (0,int(kenel_size**(1/2))):
    #         for kk in range (0,int(kenel_size**(1/2))):
    #           origin_ar_TT[i][j][int(kenel_size**(1/2))-1-k][int(kenel_size**(1/2))-1-kk] = origin_ar_T[i][j][k][kk]

    origin_ar_T = np.repeat(zero,Filter_Num_T*In_Channel_Num_T*kenel_size).reshape(Filter_Num,In_Channel_Num,kenel_size)
    origin_ar = origin_ar.reshape(Filter_Num_T,In_Channel_Num_T,kenel_size)
    origin_ar_T = origin_ar.transpose(1,0,2)
    origin_ar_TT = origin_ar_T[:,:,::-1]
    # origin_ar_TT = origin_ar_T

    # if (kenel_size==1) :
    #     origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,1) 
    #     zero = '0000'
    #     zero_ar = np.repeat(zero,6)
    #     zero_ar = zero_ar.reshape(6)
    #     temp_ar = np.repeat(zero, D).reshape(A,2,4,B,4,9)
    #     for i in range (0,A):
    #         for j in range (0,2):
    #             for k in range (0,4):
    #                 for l in range (0,B):
    #                     for m in range (0,4):
    #                         temp_ar[i][j][k][l][m] = np.concatenate( (zero_ar,origin_ar_TT[i][j][k][l][m],origin_ar_TT[i][j][k][l][m],origin_ar_TT[i][j][k][l][m]), axis = 0)
    #     # concat 4 in_channel
    #     filter_ar = np.repeat(zero, D).reshape(A,2,4,B,9,4)
    #     for i in range (0,A):
    #         for j in range (0,2):
    #             for k in range (0,4):
    #                 for l in range (0,B):
    #                     filter_ar[i][j][k][l] = temp_ar[i][j][k][l].T
    #     # print('aaaa')
    # elif (kenel_size==9) :
    #     origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,9) # Using "2" make 8 outch
    #     zero = '0000'
    #     #concat 4 in channel
    #     filter_ar = np.repeat(zero, origin_size).reshape(A,2,4,B,9,4)
    #     for fn in range (0,A):
    #         for fc2 in range (0,2):
    #             for fc1 in range (0,4):
    #                 for incn in range (0,B):
    #                     filter_ar[fn][fc2][fc1][incn] = origin_ar_TT[fn][fc2][fc1][incn].T
    # print('ssss')
    if (kenel_size==1) :
        origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,1) #(filter/8 ,2(아래로 반복), 4(4개있음), input channel/4, 4(4개를 가져감), kernel)
        zero_ar = np.repeat(zero,A*2*4*B*4*6).reshape(A,2,4,B,4,6)
        temp_ar = np.repeat(zero, D).reshape(A,2,4,B,4,9)
        temp_ar = np.concatenate( (zero_ar,origin_ar_TT,origin_ar_TT,origin_ar_TT) , axis = 5)
        # concat 4 in_channel
        filter_ar = np.repeat(zero, D).reshape(A,2,4,B,9,4)
        filter_ar = temp_ar.transpose(0,1,2,3,5,4)
    elif (kenel_size==9) :
        origin_ar_TT = origin_ar_TT.reshape(A,2,4,B,4,9) # Using "2" make 8 outch
        #concat 4 in channel
        filter_ar = np.repeat(zero, origin_size).reshape(A,2,4,B,9,4)
        filter_ar = origin_ar_TT.transpose(0,1,2,3,5,4)


    # to concat 4 filter
    # filter_ar2 = np.repeat(zero, D).reshape(A,2,B,9,16)
    # for fn in range(0,A):
    #     for fin in range (0,2):
    #         for cn in range (0,B):
    #             for d in range (0,9):
    #                 filter_ar2[fn][fin][cn][d] = np.concatenate( (filter_ar[fn][fin][0][cn][d],filter_ar[fn][fin][1][cn][d],filter_ar[fn][fin][2][cn][d],filter_ar[fn][fin][3][cn][d]), axis= 0)

    filter_ar2 = np.repeat(zero, D).reshape(A,2,B,9,16)

    filter_ar_0 = filter_ar[:,:,0,:,:,:]
    filter_ar_1 = filter_ar[:,:,1,:,:,:]
    filter_ar_2 = filter_ar[:,:,2,:,:,:]
    filter_ar_3 = filter_ar[:,:,3,:,:,:]

    filter_ar2 = np.concatenate((filter_ar_0, filter_ar_1, filter_ar_2, filter_ar_3), axis= 4)

    filter_ar3 = np.repeat(zero, D).reshape(A,B,2,9,16)  # to concat filter twice
    filter_ar3 = filter_ar2.transpose(0,2,1,3,4)

    # Wconvert_ar = filter_ar3.reshape(C,16)

    space = np.repeat(zero,1).reshape(1,1)
    vector_array = np.repeat(zero,B*16).reshape(int(B/2),2,16) #inchannel/8,2,16


    for i in range(0,int(B/2)): # filter_num / 8
        vector_array[i][0] = np.concatenate( (bias_active_ar[i*8],bias_active_ar[i*8+1],bias_active_ar[i*8+2],bias_active_ar[i*8+3],   batch_add_ar[i*8],batch_add_ar[i*8+1],batch_add_ar[i*8+2],batch_add_ar[i*8+3],
                                                bias_active_ar[i*8+4],bias_active_ar[i*8+5],bias_active_ar[i*8+6],bias_active_ar[i*8+7], batch_add_ar[i*8+4],batch_add_ar[i*8+5],batch_add_ar[i*8+6],batch_add_ar[i*8+7] ),axis=0)
        vector_array[i][1] = np.concatenate( (batch_mul_ar[i*8],batch_mul_ar[i*8+1],batch_mul_ar[i*8+2],batch_mul_ar[i*8+3],   space[0],space[0],space[0],space[0],
                                            batch_mul_ar[i*8+4],batch_mul_ar[i*8+5],batch_mul_ar[i*8+6],batch_mul_ar[i*8+7], space[0],space[0],space[0],space[0] ),axis=0)


    # vector_array = vector_array.reshape(B,16)
    # zero = '0000'

    Wconvert_array = filter_ar3.reshape(A,B,18,16) #fil/8, in/4
    # biacsc_ar = np.repeat(zero,B*16).reshape(int(B/2),1,2,16)
    biacsc_ar = vector_array.reshape(int(B/2),1,2,16)
    concat_ar = np.repeat(zero,G).reshape(int(A/2),B*2,20,16)

    for k in range (0,int(A/2)):
        for i in range (0,B):
            concat_ar[k][2*i]   = np.concatenate((Wconvert_array[2*k][i],biacsc_ar[int(i/4)*2][0]), axis=0)
            concat_ar[k][2*i+1] = np.concatenate((Wconvert_array[2*k+1][i],biacsc_ar[int(i/4)*2+1][0]), axis=0)

    convert_ar = np.repeat(zero,G).reshape(H,16)
    convert_ar = concat_ar.reshape(H,16)


    Weight_List = []
    Weight_List.clear()
    for value in convert_ar:
        Result = ''.join(value)
        Weight_List.append(Result)

    df = pd.DataFrame(Weight_List)
    df1, df2 = Separated_Weight_DDR_Channel(df)
    #   df1.to_csv(Write_Path_Ch0, index=False, header=False, sep='\t')
    #   df2.to_csv(Write_Path_Ch1, index=False, header=False, sep='\t')
    Weight_OtherLayer_Channel0 = df1.values.tolist()
    Weight_OtherLayer_Channel1 = df2.values.tolist()
    Weight_OtherLayer = [Weight_OtherLayer_Channel0, Weight_OtherLayer_Channel1]
    return Weight_OtherLayer


# def Fmap_Ordering(Channel, Data_List):
    origin = pd.DataFrame(Data_List)
    Output_Channel = Channel
    origin_ar = np.array(origin)
    origin_size = np.size(origin_ar)
    Fmap_size = int((origin_size/Output_Channel)**(1/2))

    zero = "0000"

    # original fmap shape
    origin_ar = origin_ar.reshape(Output_Channel,Fmap_size,Fmap_size)

    # change row and col each other(전치)
    for i in range (0,Output_Channel):
        origin_ar[i] = origin_ar[i].T

    origin_ar = origin_ar.reshape(Output_Channel,Fmap_size,Fmap_size)
    iter_13 = int(Fmap_size/13)

    concat_ar = np.repeat(zero, Output_Channel*Fmap_size*(Fmap_size+iter_13*3)).reshape(Output_Channel,(Fmap_size+iter_13*3),Fmap_size)

    for i in range (0,Output_Channel):
        for j in range (0,iter_13):
            for k in range (0,Fmap_size):
                concat_ar[i][j*16+0][k]  = origin_ar[i][j*13+0][k]
                concat_ar[i][j*16+1][k]  = origin_ar[i][j*13+1][k]
                concat_ar[i][j*16+2][k]  = origin_ar[i][j*13+2][k]
                concat_ar[i][j*16+3][k]  = origin_ar[i][j*13+3][k]
                concat_ar[i][j*16+4][k]  = origin_ar[i][j*13+4][k]
                concat_ar[i][j*16+5][k]  = origin_ar[i][j*13+5][k]
                concat_ar[i][j*16+6][k]  = origin_ar[i][j*13+6][k]
                concat_ar[i][j*16+7][k]  = origin_ar[i][j*13+7][k]
                concat_ar[i][j*16+8][k]  = origin_ar[i][j*13+8][k]
                concat_ar[i][j*16+9][k]  = origin_ar[i][j*13+9][k]
                concat_ar[i][j*16+10][k] = origin_ar[i][j*13+10][k]
                concat_ar[i][j*16+11][k] = origin_ar[i][j*13+11][k]
                concat_ar[i][j*16+12][k] = origin_ar[i][j*13+12][k]
                concat_ar[i][j*16+13][k] = origin_ar[i][j*13+12][k]
                concat_ar[i][j*16+14][k] = origin_ar[i][j*13+12][k]
                concat_ar[i][j*16+15][k] = origin_ar[i][j*13+12][k]
    concat_ar = concat_ar.reshape(Output_Channel*(Fmap_size+iter_13*3),Fmap_size,1)


    four_ar1 = np.repeat(zero,int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/4)).reshape(int(Output_Channel*(Fmap_size+iter_13*3)/4),Fmap_size,1)
    four_ar2 = np.repeat(zero,int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/4)).reshape(int(Output_Channel*(Fmap_size+iter_13*3)/4),Fmap_size,1)
    four_ar3 = np.repeat(zero,int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/4)).reshape(int(Output_Channel*(Fmap_size+iter_13*3)/4),Fmap_size,1)
    four_ar4 = np.repeat(zero,int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/4)).reshape(int(Output_Channel*(Fmap_size+iter_13*3)/4),Fmap_size,1)

    for i in range (0,int(Output_Channel*(Fmap_size+iter_13*3)/4)):
        for j in range (0,Fmap_size):
            four_ar1[i][j][0] = concat_ar[4*i][j][0]
            four_ar2[i][j][0] = concat_ar[4*i+1][j][0]
            four_ar3[i][j][0] = concat_ar[4*i+2][j][0]
            four_ar4[i][j][0] = concat_ar[4*i+3][j][0]

    four_ar1 = four_ar1.reshape(Output_Channel,int(Fmap_size*(Fmap_size+iter_13*3)/4),1)
    four_ar2 = four_ar2.reshape(Output_Channel,int(Fmap_size*(Fmap_size+iter_13*3)/4),1)
    four_ar3 = four_ar3.reshape(Output_Channel,int(Fmap_size*(Fmap_size+iter_13*3)/4),1)
    four_ar4 = four_ar4.reshape(Output_Channel,int(Fmap_size*(Fmap_size+iter_13*3)/4),1)

    fmap1 = np.repeat(zero,int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/4)).reshape(int(Output_Channel/16),int(Fmap_size*(Fmap_size+iter_13*3)/4),16)
    fmap2 = np.repeat(zero,int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/4)).reshape(int(Output_Channel/16),int(Fmap_size*(Fmap_size+iter_13*3)/4),16)
    fmap3 = np.repeat(zero,int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/4)).reshape(int(Output_Channel/16),int(Fmap_size*(Fmap_size+iter_13*3)/4),16)
    fmap4 = np.repeat(zero,int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/4)).reshape(int(Output_Channel/16),int(Fmap_size*(Fmap_size+iter_13*3)/4),16)

    # concat 4 input channel
    for i in range (0, int(Output_Channel/16)):
        fmap1[i] = np.concatenate( ( four_ar1[16*i+0], four_ar1[16*i+1], four_ar1[16*i+2], four_ar1[16*i+3], four_ar2[16*i+0], four_ar2[16*i+1], four_ar2[16*i+2], four_ar2[16*i+3], four_ar3[16*i+0], four_ar3[16*i+1], four_ar3[16*i+2], four_ar3[16*i+3], four_ar4[16*i+0], four_ar4[16*i+1], four_ar4[16*i+2], four_ar4[16*i+3]), axis=1)
        fmap2[i] = np.concatenate( ( four_ar1[16*i+4], four_ar1[16*i+5], four_ar1[16*i+6], four_ar1[16*i+7], four_ar2[16*i+4], four_ar2[16*i+5], four_ar2[16*i+6], four_ar2[16*i+7], four_ar3[16*i+4], four_ar3[16*i+5], four_ar3[16*i+6], four_ar3[16*i+7], four_ar4[16*i+4], four_ar4[16*i+5], four_ar4[16*i+6], four_ar4[16*i+7]), axis=1)
        fmap3[i] = np.concatenate( ( four_ar1[16*i+8], four_ar1[16*i+9], four_ar1[16*i+10], four_ar1[16*i+11], four_ar2[16*i+8], four_ar2[16*i+9], four_ar2[16*i+10], four_ar2[16*i+11], four_ar3[16*i+8], four_ar3[16*i+9], four_ar3[16*i+10], four_ar3[16*i+11], four_ar4[16*i+8], four_ar4[16*i+9], four_ar4[16*i+10], four_ar4[16*i+11]), axis=1)
        fmap4[i] = np.concatenate( ( four_ar1[16*i+12], four_ar1[16*i+13], four_ar1[16*i+14], four_ar1[16*i+15], four_ar2[16*i+12], four_ar2[16*i+13], four_ar2[16*i+14], four_ar2[16*i+15], four_ar3[16*i+12], four_ar3[16*i+13], four_ar3[16*i+14], four_ar3[16*i+15], four_ar4[16*i+12], four_ar4[16*i+13], four_ar4[16*i+14], four_ar4[16*i+15]), axis=1)

    # print(origin_size)
    # print(fmap1.shape)

    # # concat 4 pixel
    # convert_ar = np.repeat(zero,origin_size).reshape(int(origin_size/16),16)
    # for i in range (0,int(Output_Channel/16)) :
    #   for ch in range (0,16): # 16 고정
    #     for pix in range (0,int(Fmap_size*Fmap_size/4)):
    #       convert_ar [i*Fmap_size*Fmap_size+pix*4][ch]= fmap1.reshape(int(Output_Channel/4),int(Fmap_size*Fmap_size/4),16)[i*4+0][pix][ch]
    #       convert_ar [i*Fmap_size*Fmap_size+pix*4+1][ch]= fmap2.reshape(int(Output_Channel/4),int(Fmap_size*Fmap_size/4),16)[i*4+1][pix][ch]
    #       convert_ar [i*Fmap_size*Fmap_size+pix*4+2][ch]= fmap3.reshape(int(Output_Channel/4),int(Fmap_size*Fmap_size/4),16)[i*4+2][pix][ch]
    #       convert_ar [i*Fmap_size*Fmap_size+pix*4+3][ch]= fmap4.reshape(int(Output_Channel/4),int(Fmap_size*Fmap_size/4),16)[i*4+3][pix][ch]

    trans_ar = np.repeat(zero,Output_Channel*Fmap_size*(Fmap_size+iter_13*3)).reshape(int(Output_Channel*Fmap_size*(Fmap_size+iter_13*3)/16),16)
    for i in range (0,int(Output_Channel/16)):
        for ch in range (0,16):
            for pix in range (0,int(Fmap_size*(Fmap_size+iter_13*3)/4)):
                trans_ar [i*Fmap_size*(Fmap_size+iter_13*3)+pix*4][ch]= fmap1[i][pix][ch]
                trans_ar [i*Fmap_size*(Fmap_size+iter_13*3)+pix*4+1][ch]= fmap2[i][pix][ch]
                trans_ar [i*Fmap_size*(Fmap_size+iter_13*3)+pix*4+2][ch]= fmap3[i][pix][ch]
                trans_ar [i*Fmap_size*(Fmap_size+iter_13*3)+pix*4+3][ch]= fmap4[i][pix][ch]

    Fmap_List = []
    Fmap_List.clear()
    for value in trans_ar:
        Result = ''.join(value)
        Fmap_List.append(Result)

    df = pd.DataFrame(Fmap_List)
    df1, df2 = Separated_Fmap_DDR_Channel(df)
#   df1.to_csv(Write_Path_Ch0, index=False, header=False, sep='\t')
#   df2.to_csv(Write_Path_Ch1, index=False, header=False, sep='\t')
    Image_OtherLayer_Channel0 = df1.values.tolist()
    Image_OtherLayer_Channel1 = df2.values.tolist()
    Image_OtherLayer = [Image_OtherLayer_Channel0, Image_OtherLayer_Channel1]
    return Image_OtherLayer




def Fmap_Ordering(Channel, Data_List):
    Output_Channel = Channel
    origin = pd.DataFrame(Data_List)
    origin_ar = np.array(origin)
    origin_size = np.size(origin_ar)
    Fmap_size = int((origin_size/Output_Channel)**(1/2))
    iter_13 = Fmap_size//13

    # original fmap shape
    origin_ar = origin_ar.reshape(Output_Channel,Fmap_size,Fmap_size).transpose(0,2,1)
    origin_ar = origin_ar.reshape(Output_Channel*iter_13,13,Fmap_size)
    origin_ar = np.concatenate( (origin_ar,origin_ar[:,12:13],origin_ar[:,12:13],origin_ar[:,12:13]), axis=1 ).reshape(Output_Channel//4,4, iter_13*4,4, Fmap_size) #(Output_Channel, iter_13*16, Fmap_size)
    origin_ar = origin_ar.transpose(0,2,4,3,1) #(Output_Channel//4, iter_13*4, Fmap_size, 4(iter_13*4), 4(Output_Channel//16))
    origin_ar = origin_ar.reshape(Output_Channel//16,4, iter_13*4*Fmap_size, 16)

    final_ar1 = np.concatenate((origin_ar[:,0],origin_ar[:,1]), axis=2).reshape(Output_Channel*iter_13*Fmap_size//2, 16)
    final_ar2 = np.concatenate((origin_ar[:,2],origin_ar[:,3]), axis=2).reshape(Output_Channel*iter_13*Fmap_size//2, 16)

    final_ar = np.concatenate((final_ar1,final_ar2), axis=0)

    final_list = []
    final_list.clear()
    for value in final_ar:
        Result = ''.join(value)
        final_list.append(Result)

    df = pd.DataFrame(final_list)
    final_ar = np.array(df).reshape(2,Output_Channel*iter_13*Fmap_size//2)
    df1 = pd.DataFrame(final_ar[0])
    df2 = pd.DataFrame(final_ar[1])

    Image_OtherLayer_Channel0 = df1.values.tolist()
    Image_OtherLayer_Channel1 = df2.values.tolist()

    Image_OtherLayer = [Image_OtherLayer_Channel0, Image_OtherLayer_Channel1]
    return Image_OtherLayer


def Weight_Bfloat16(Input_List, Exponent_Bit, Mantissa_Bit):
    # Convert all the Weight Net List into Bfloat16
    
    Weight_Bfloat16_List = []
    Weight_Bfloat16_List.clear()
    
    for sublist in Input_List:
        converted_sublist = []  # Create a new sublist for converted values
        converted_sublist.clear() 
        for Value in sublist:
            Binary_Value = Floating2Binary(Value, Exponent_Bit, Mantissa_Bit)
            Hexadecimal_Value = hex(int(Binary_Value, 2))[2:]
            Truncated_Rounded_Hex = Truncating_Rounding(Hexadecimal_Value)
            converted_sublist.append(Truncated_Rounded_Hex)
        Weight_Bfloat16_List.append(converted_sublist)
        # print("sublist : ", len(sublist))

    return Weight_Bfloat16_List

def Weight_FP32(Input_List, Exponent_Bit, Mantissa_Bit):
    # Convert all the Weight Net List into FP32
    
    Weight_Bfloat16_List = []
    Weight_Bfloat16_List.clear()
    
    for sublist in Input_List:
        converted_sublist = []  # Create a new sublist for converted values
        for Value in sublist:
            Binary_Value = Floating2Binary(Value, Exponent_Bit, Mantissa_Bit)
            Hexadecimal_Value = hex(int(Binary_Value, 2))[2:].upper()
            converted_sublist.append(Hexadecimal_Value)
        Weight_Bfloat16_List.append(converted_sublist)
    converted_sublist.clear()
    return Weight_Bfloat16_List

def Bias_Bfloat16(Input_List, Exponent_Bit, Mantissa_Bit):
    # Convert Bias list into Bfloat16
    Bias_List = []
    for Value in Input_List:
        Binary_Value = Floating2Binary(Value, Exponent_Bit, Mantissa_Bit)
        Hexadecimal_Value = hex(int(Binary_Value, 2))[2:]
        Truncated_Rounded_Hex = Truncating_Rounding(Hexadecimal_Value)
        Bias_List.append(Truncated_Rounded_Hex)   
    return Bias_List

def Bias_FP32(Input_List, Exponent_Bit, Mantissa_Bit):
    # Convert Bias list into FP32
    
    Bias_List = []
    Bias_List.clear()
    for Value in Input_List:
        Binary_Value = Floating2Binary(Value, Exponent_Bit, Mantissa_Bit)
        Hexadecimal_Value = hex(int(Binary_Value, 2))[2:].upper()
        Bias_List.append(Hexadecimal_Value)   
    return Bias_List


def BN_Bfloat16(Input_List, Exponent_Bit, Mantissa_Bit):
    # Convert all the Weight Net List into Bfloat16
    
    BN_Bfloat16_List = []
    BN_Bfloat16_List.clear()
    for sublist in Input_List:
        converted_sublist = []  # Create a new sublist for converted values
        for Value in sublist:
            Binary_Value = Floating2Binary(Value, Exponent_Bit, Mantissa_Bit)
            Hexadecimal_Value = hex(int(Binary_Value, 2))[2:]
            Truncated_Rounded_Hex = Truncating_Rounding(Hexadecimal_Value)
            converted_sublist.append(Truncated_Rounded_Hex)
        BN_Bfloat16_List.append(converted_sublist)

    return BN_Bfloat16_List

def BN_FP32(Input_List, Exponent_Bit, Mantissa_Bit):
    # Convert all the BN Net List into FP32
    
    BN_FP32_List = []
    BN_FP32_List.clear()
    for sublist in Input_List:
        converted_sublist = []  # Create a new sublist for converted values
        for Value in sublist:
            Binary_Value = Floating2Binary(Value, Exponent_Bit, Mantissa_Bit)
            Hexadecimal_Value = hex(int(Binary_Value, 2))[2:].upper()
            converted_sublist.append(Hexadecimal_Value)
        BN_FP32_List.append(converted_sublist)

    return BN_FP32_List

def Read_BN(File_List, Read_Folder_Path):
    # Read all the Bias from a Bias_Folder
    
    BN_List = []
    BN_List.clear()
    List_Sorted = sorted(File_List)  # Sort the file names alphabetically

    for i, file in enumerate(List_Sorted):
        Read_File_Path = os.path.join(Read_Folder_Path, file)

        with open(Read_File_Path, mode="r") as file_r:
            Input = file_r.read()

        Input_List = [np.float32(Value) for Value in Input.split()]
        BN_List.append(Input_List)
        
    return BN_List

def Weight_Gradient_Hardware_ReOrdering_Layer0(Out_CH, In_CH, DataCH0_List, DataCH1_List):
  
    Filter_Num = Out_CH
    In_Channel_Num = In_CH

    origin0 = pd.DataFrame(DataCH0_List)
    origin1 = pd.DataFrame(DataCH1_List)

    origin_ar0 = np.array(origin0)
    origin_ar1 = np.array(origin1)
    half_size = np.size(origin_ar0)//16
    A = int(half_size*16)
    kenel_size = int(half_size/(Filter_Num*In_Channel_Num/16/2))

    convert_ar0 = origin_ar0.reshape(half_size//kenel_size,kenel_size,16)
    convert_ar1 = origin_ar1.reshape(half_size//kenel_size,kenel_size,16)

    convert_ar0 = convert_ar0.transpose(0,2,1).reshape(Filter_Num*In_Channel_Num//256,2,16,4,kenel_size)
    convert_ar1 = convert_ar1.transpose(0,2,1).reshape(Filter_Num*In_Channel_Num//256,2,16,4,kenel_size)

    concat_ar = np.concatenate( (convert_ar0[:,0], convert_ar0[:,1], convert_ar1[:,0], convert_ar1[:,1]), axis=2 ) #shape(Filter_Num*In_Channel_Num//256,16,16,kenel_size)
    concat_ar = concat_ar.reshape(Filter_Num//16,In_Channel_Num//16,16,16,kenel_size)

    final_ar = concat_ar.transpose(0,2,1,3,4).reshape(Filter_Num,In_Channel_Num,kenel_size)
    
    final_ar1 = final_ar[:,0:3]

    final_ar1 = final_ar1.reshape(-1)

    # Reshape convert_ar
    # convert_ar = convert_ar.reshape(Filter_Num * In_Channel_Num * kenel_size)
    
    
    Weight_Gradient_List = []
    Weight_Gradient_List.clear()
    for value in final_ar1:
        Result = ''.join(value)
        Weight_Gradient_List.append(Result)
    #   df = pd.DataFrame(Weight_Gradient_List)
    #   df.to_csv(Write_Path,index=False, header=False, sep='\t')
    return Weight_Gradient_List
