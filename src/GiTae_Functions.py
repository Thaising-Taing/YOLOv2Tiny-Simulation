import torch
from torch import optim
import pdb
import tkinter
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
from pypcie import Device
from ast import literal_eval
import subprocess

import warnings
warnings.filterwarnings("ignore")
from functools import lru_cache
import numba

from tkinter.constants import DISABLED, NORMAL 

import os
import sys
sys.path.append("../")
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),"Dataset"))
sys.path.append(os.path.join(os.getcwd(),"src"))
sys.path.append(os.path.join(os.getcwd(),"src/Main_Processing_Scratch"))
sys.path.append(os.path.join(os.getcwd(),"src/Pre_Processing_Scratch"))
sys.path.append(os.path.join(os.getcwd(),"src/Post_Processing_Scratch"))
sys.path.append(os.path.join(os.getcwd(),"src/Weight_Update_Algorithm"))
sys.path.append(os.path.join(os.getcwd(),"Codes"))
sys.path.append("/home/msis/Desktop/pcie_python/GUI")
from  XdmaAccess import XdmaAccess
from Pre_Processing_Scratch.Pre_Processing import *
from Pre_Processing_Scratch.Pre_Processing_Function import *
from Post_Processing_Scratch.Post_Processing_2Iterations import Post_Processing
from Pre_Processing_Scratch.ImageLoader import ImageLoader
import time
from tabulate import tabulate
import os.path 
import matplotlib.pyplot as plt
from Dataset.roidb import RoiDataset, detection_collate
from Dataset.factory import get_imdb
from torch.utils.data import DataLoader
import pickle
from Post_Processing_Scratch.Post_Processing_2Iterations_Training_Inference import *
from Detection.Detection import *

from Weight_Update_Algorithm.weight_update import *
from Weight_Update_Algorithm.yolov2_tiny import *

from GiTae_Functions import *

MAX_LINE_LENGTH = 1000
DEBUG = False
DEBUG2 = True

def save_txt(fname, data, module=[], layer_no=[], save_txt=False, save_hex=False, phase=[]):
    # if DEBUG: print(f"Type of data: {type(data)}")
    if save_txt or save_hex:
        if type(data) is dict:
            for _key in data.keys():
                _fname = fname + f'_{_key}'
                save_file(_fname, data[_key])

        else:
            if module == [] and layer_no == []:
                Out_Path = f'Outputs_Torch/{os.path.split(fname)[0]}'
                fname = os.path.split(fname)[1]
            else:
                Out_Path = f'Outputs_Torch/By_Layer/'
                if layer_no != []: Out_Path += f'Layer{layer_no}/'
                if module != []: Out_Path += f'{module}/'
                if phase != []: Out_Path += f'{phase}/'
                fname = fname

            if save_txt: filename = os.path.join(Out_Path, fname + '.txt')
            # if save_hex: hexname = os.path.join(Out_Path, fname + '_hex.txt')

            Path(Out_Path).mkdir(parents=True, exist_ok=True)

            if torch.is_tensor(data):
                try:
                    data = data.detach()
                except:
                    pass
                data = data.numpy()

            if save_txt: outfile = open(filename, mode='w')
            if save_txt: outfile.write(f'{data.shape}\n')

            # if save_hex: hexfile = open(hexname, mode='w')
            # if save_hex: hexfile.write(f'{data.shape}\n')

            if len(data.shape) == 0:
                if save_txt: outfile.write(f'{data}\n')
                # if save_hex: hexfile.write(f'{data}\n')
                pass
            elif len(data.shape) == 1:
                for x in data:
                    if save_txt: outfile.write(f'{x}\n')
                    # if save_hex: hexfile.write(f'{convert_to_hex(x)}\n')
                    pass
            else:
                w, x, y, z = data.shape
                # if w != 0:
                #     Out_Path += f'img{w+1}'
                for _i in range(w):
                    for _j in range(x):
                        for _k in range(y):
                            for _l in range(z):
                                _value = data[_i, _j, _k, _l]
                                if save_txt: outfile.write(f'{_value}\n')
                                # if save_hex: hexfile.write(f'{convert_to_hex(_value)}\n')
                                pass

            # if save_hex: hexfile.close()
            if save_txt: outfile.close()

            if DEBUG: print(f'\t\t--> Saved {filename}')
            # if save_hex: if DEBUG: print(f'\t\t--> Saved {hexname}')

def Save_File(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)   

def Debug_With_Slave():
    d = Device("0000:08:00.0")
    bar = d.bar[0]

    data_read = open("result/slave_result.txt", mode="w+")
    i=0
    for i in range(0,16): 
        Read_Data = bar.read(0X00 + (i*4))
        data_read.write(str(Read_Data) + "\n") 

def Read_DDR(Rd_Address, End_Address):
    device = XdmaAccess(0)
    Read_Data_List = device.read_dma(Rd_Address, ((End_Address-Rd_Address)))
    # if DEBUG: print("Read_Data_List : ", Read_Data_List)
    device.__close__()
    return Read_Data_List  

def Write_DDR(Wr_Data_List, Wr_Address):
    device = XdmaAccess(0)
    data_to_write = []
    for line in Wr_Data_List:
        line_data = int(line.strip(), 16)
        data_to_write.append(line_data)
    data_to_write_array = np.array(data_to_write, dtype=np.uint32)
    device.write_dma(Wr_Address, data_to_write_array) 
    device.__close__()

def Microcode(read_path):
    Microcode_List = []
    Microcode_List.clear()
    read = open(read_path, mode="r")
    Microcode = read.readlines()
    for value in Microcode:
        value = value.replace(',', '').replace('\n', '')
        value = int(value, 16)
        Microcode_List.append(value)
    return Microcode_List 
 
def fill_0_data(data):
    data_str = str(data).zfill(8)
    return data_str

def flip_lines(lines):
    flipped_lines = lines[::-1]
    return flipped_lines

def process_input_lines(lines):
    processed_lines = []
    processed_lines.clear()

    for i in range(0, len(lines), 8):
        chunk = lines[i:i+8] 

        hex_lines = [fill_0_data(hex(line)[2:]) for line in chunk]

        flipped_lines = flip_lines(hex_lines)
        combined_line = ''.join(flipped_lines)

        processed_lines.append(combined_line)

    return processed_lines

def flip_8_line(data):
    for i in range(0, len(data), 8):
        group = data[i:i+8]
        yield from reversed(group)

def data_256_32(data_list):
    output_list = []
    output_list.clear()
    Hex32 = []
    Hex32.clear()
    flip_data_list = []
    flip_data_list.clear()

    for data in data_list:
        for value in data:
            output = [value[i:i+8] for i in range(0, len(value), 8)]
            output_list += output
            # output_list.append(output)  
    # for segments in output_list:
    #     for segment in segments:
    #         Hex32.append(segment)
         
    flip_data = flip_8_line(output_list)
    flip_data_list.extend(flip_data) 

    return flip_data_list

def data_32_to_16(data):
    data = flip_8_line(data)
    
    hex_values = (hex(value)[2:].upper().zfill(8) for value in data)
    hex_strings = ''.join(hex_values)
    formatted_data = [hex_strings[i:i+4] for i in range(0, len(hex_strings), 4)]
    return formatted_data


def clean_string(text):
    if isinstance(text, str):
        return text.replace('[', '').replace(']', '').replace("'", '')
    return text

def backward_LightNorm(grad_output, cache):
    X, gamma, beta, output, scale, scale_fix, avg, avg_max, avg_min, eps, num_chunks, max_index, min_index = cache
    B, C, H, W = X.shape
    dL_dxi_hat = grad_output * gamma.view(1, -1, 1, 1)
    
    # Compute dL_dvar
    dL_dvar = (dL_dxi_hat * (X - avg) * -0.5 * torch.sqrt(scale) * torch.sqrt(scale) * torch.sqrt(scale)).sum(dim=(0, 2, 3), keepdim=True)
    
    # Compute dL_dxmax_mean and dL_dxmin_mean
    dL_dxmax_mean = (dL_dvar / scale_fix).sum(dim=(0, 2, 3), keepdim=True)
    dL_dxmin_mean = (-1 * dL_dvar / scale_fix).sum(dim=(0, 2, 3), keepdim=True)
    
    # Compute dL_dxmax and dL_dxmin
    dL_dxmax = (dL_dxmax_mean / num_chunks).sum(dim=(0, 2, 3), keepdim=True)
    dL_dxmin = (dL_dxmin_mean / num_chunks).sum(dim=(0, 2, 3), keepdim=True)
    
    # Compute dL_dgamma and dL_dbeta
    dL_dgamma = (grad_output * output).sum(dim=(0, 2, 3), keepdim=True)
    dL_dbeta = grad_output.sum(dim=(0, 2, 3), keepdim=True)
    dL_davg = grad_output.sum(dim=(0, 2, 3), keepdim=True)

    # Average per channel
    avg_pc = (dL_dxi_hat * -1.0).sum(dim=(0, 2, 3), keepdim=True) / (B * H * W)
    dL_dxi_ = avg_pc + dL_dxi_hat
    
    # Backward coefficient
    backward_const = scale
    
    # Final output calculation
    dL_dxi = dL_dxi_ * backward_const

    return dL_dgamma, dL_dbeta, avg_pc, backward_const   

def split_location(mask_location): 
    relu_mask = torch.zeros_like(mask_location)
    relu_mask[mask_location>3] = 1
    
    location = torch.zeros_like(mask_location)
    location[mask_location==0] = 0
    location[mask_location==2] = 1
    location[mask_location==1] = 2
    location[mask_location==3] = 3
    location[mask_location==4] = 0
    location[mask_location==6] = 1
    location[mask_location==5] = 2
    location[mask_location==7] = 3
    
    return relu_mask, location
    
def backward_active(gradient, relu_mask, alpha=0.1):
    dx, x = None, relu_mask
    
    dl = torch.ones_like(x)
    dl[x > 0] = alpha
    dx = gradient * dl
    
    return dx

def backward_ReLU(dout, cache, alpha=0.1):
    dx, x = None, cache
    
    dl = torch.ones_like(x)
    dl[x < 0] = alpha
    dx = dout * dl
    
    return dx

def backward_MaxPool(dout, x, layer_no=[], save_txt=False, save_hex=False, phase=[]):
    
    x = x
    dx = None

    N, C, H, W = x.shape
    stride = 2
    pool_width = 2
    pool_height = 2
    
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
    dx = torch.zeros_like(x)
    
    backward_positions = []
    backward_positions.clear()
    
    
    for n in range(N):
        for c in range(C):
            temp_positions = []
            temp_positions.clear()
            for height in range(H_out):
                for width in range(W_out):
                    local_x = x[n, c, height * stride:height * stride + pool_height,
                            width * stride:width * stride + pool_width]
                    
                    shape_local_x = local_x.shape
                    
                    input_tensor = local_x.reshape(-1)
                    
                    # if DEBUG: print("input_tensor",input_tensor)
                    
                    # if DEBUG: print("input_tensor", input_tensor.shape, input_tensor)
                    local_dw = torch.zeros_like(input_tensor)

                    max_index = torch.argmax(input_tensor)
                    
                    max_value = input_tensor[max_index]
                    
                    all_max_indices = torch.nonzero(input_tensor == max_value).flatten()
                    
                    
                    last_index_of_highest_value = all_max_indices[-1].item()
                    backward_positions.append(last_index_of_highest_value)
                    temp_positions.append(last_index_of_highest_value)
                    # values, indicies = input_tensor.max(-1)
                    local_dw[last_index_of_highest_value] = dout[n, c, height, width]
                    dx[n, c, height * stride:height * stride + pool_height,
                    width * stride:width * stride + pool_width] = local_dw.reshape(shape_local_x)
    
    backward_positions = torch.tensor(backward_positions)

    return dx

def backward_MaxPool_Location(dout, Location, layer_no=[], save_txt=False, save_hex=False, phase=[]):
    
    # x, pool_param = cache

    dx = None

    N, C, H, W = Location.shape
    H = 2*H
    W = 2*W
    # stride = pool_param['stride']
    # pool_width = pool_param['pool_width']
    # pool_height = pool_param['pool_height']
    stride = 2
    pool_width = 2
    pool_height = 2
    
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
    dx = torch.zeros(N, C, H, W)
    
    for n in range(N):
        for c in range(C):
            temp_positions = []
            for height in range(H_out):
                for width in range(W_out):
                    local_dw = torch.zeros(4)
                    local_dw[int(Location[n, c, height, width])] = dout[n, c, height, width]
                    dx[n, c, height * stride:height * stride + pool_height,
                    width * stride:width * stride + pool_width] = local_dw.reshape(2,2)
    
    return dx    

def check_irq_layer0():
    input_file_name = "/proc/interrupts"
    output_file_name_1 = "src/GiTae/interrupt.txt"
    output_file_name_2 = "src/GiTae/interrupt_old.txt"
    irq_val=0
    while irq_val == 0:                        
        if os.path.isfile(output_file_name_2):

            with open(input_file_name, "r") as input_file, \
            open(output_file_name_1, "w") as output_file:

                for line in input_file:
                    if "xdma" in line:
                        output_file.write(line)

            input_file.close()
            output_file.close()
            
            with open(output_file_name_1, "r") as file1, \
                open(output_file_name_2, "r") as file2:
                    ch1 = file1.read(1)
                    ch2 = file2.read(1)

                    while ch1 and ch2:
                        if ch1 != ch2:
                            # if DEBUG: print("interrupt1: 1")
                            irq_val = 1
                            # self.L1_IRQ_canvas.itemconfig(self.L1_IRQ, fill="green")
                        ch1 = file1.read(1)
                        ch2 = file2.read(1)


                    # if irq_val != 1:
                    #     if DEBUG: print("layer0 interrupt1: 0")

                    with open(output_file_name_1, "rb") as file1, \
                        open(output_file_name_2, "wb") as file2:

                        buffer = file1.read(MAX_LINE_LENGTH)
                        while buffer:
                            file2.write(buffer)
                            buffer = file1.read(MAX_LINE_LENGTH)

                    # if DEBUG: print("Done")
                    file1.close()
                    file2.close()
        else:  
            with open(input_file_name, "r") as input_file, \
                open(output_file_name_1, "w") as output_file:

                for line in input_file:
                    if "xdma" in line:
                        output_file.write(line)
                        if " 1 " in line:
                            irq_val=1
                            # self.L1_IRQ_canvas.itemconfig(self.L1_IRQ, fill="green")

                        #     if DEBUG: print("interrupt: 1")
                        # else:
                        #     irq_val=0
                        #     if DEBUG: print("layer0 interrupt0: 0") 

                input_file.close()
                output_file.close()            

                if irq_val == 1:
                    with open(output_file_name_1, "rb") as file1, \
                        open(output_file_name_2, "wb") as file2:

                        buffer = file1.read(MAX_LINE_LENGTH)
                        while buffer:
                            file2.write(buffer)
                            buffer = file1.read(MAX_LINE_LENGTH)    

                    file1.close()
                    file2.close()  

def check_irq_otherlayer():
    input_file_name = "/proc/interrupts"
    output_file_name_1 = "src/GiTae/interrupt.txt"
    output_file_name_2 = "src/GiTae/interrupt_old.txt"
    irq_val=0
    while irq_val == 0:                        
        with open(input_file_name, "r") as input_file, \
            open(output_file_name_1, "w") as output_file:

            for line in input_file:
                if "xdma" in line:
                    output_file.write(line)
        input_file.close()
        output_file.close()              

        with open(output_file_name_1, "r") as file1, \
            open(output_file_name_2, "r") as file2:
                ch1 = file1.read(1)
                ch2 = file2.read(1)

                while ch1 and ch2:
                    if ch1 != ch2:
                        # if DEBUG: print("interrupt: 1")
                        irq_val = 1
                        # L1_IRQ_canvas.itemconfig(L1_IRQ, fill="green")
                    ch1 = file1.read(1)
                    ch2 = file2.read(1)

                # if irq_val != 1:
                #     if DEBUG: print("layer1 interrupt: 0")

                with open(output_file_name_1, "rb") as file1, \
                    open(output_file_name_2, "wb") as file2:

                    buffer = file1.read(MAX_LINE_LENGTH)
                    while buffer:
                        file2.write(buffer)
                        buffer = file1.read(MAX_LINE_LENGTH)

                # if DEBUG: print("Done")
                file1.close()
                file2.close()
        
        # if DEBUG: print("extract xdma line Done!\n")     

def resume():
    d = Device("0000:08:00.0")
    bar = d.bar[0]

    # resume    
    # if DEBUG: print("Resume Process")
    bar.write(0x20, 1)

    bar.write(0x20, 0)

# def data_32_to_16(data):
#     data = flip_8_line(data)
#     hex_values = (hex(value)[2:].upper().zfill(8) for value in data)
#     hex_strings = ''.join(hex_values)
#     formatted_data = [hex_strings[i:i+4] for i in range(0, len(hex_strings), 4)]
#     return formatted_data

class YOLOv2_Tiny_FPGA(object):

    def __init__(self, Weight_Dec, Bias_Dec, Beta_Dec, Gamma_Dec, Running_Mean_Dec, Running_Var_Dec, app_instance):
        self.Weight_Dec = Weight_Dec
        self.Bias_Dec = Bias_Dec
        self.Beta_Dec = Beta_Dec
        self.Gamma_Dec = Gamma_Dec
        self.Running_Mean_Dec = Running_Mean_Dec
        self.Running_Var_Dec = Running_Var_Dec
        self.Image = []
        self.app_instance = app_instance    
        self.custom_model = Yolov2()
        self.custom_optimizer = optim.SGD(self.custom_model.parameters(), lr=0.001, momentum=0.001, weight_decay=0.001)
        self.PreProcessing = app_instance.PreProcessing  
               
    def Forward(self,data):
        global layer0_cache, layer1_cache, layer2_cache, layer3_cache, layer4_cache, layer5_cache, layer6_cache, layer7_cache     
        start = time.time()
        #################################################
        #                Layer 0 Start                  #
        #################################################       
        # layer0 capture interrupt
        # TODO: Moved check_irq func to main
        # check_irq_layer0() 
        # self.app_instance.change_color(self.app_instance.L1_IRQ_canvas, self.app_instance.L1_IRQ, "green")
        global OutImage_1st_Layer0, OutImage_1st_Layer1, OutImage_1st_Layer2, OutImage_1st_Layer3, OutImage_1st_Layer4,\
        OutImage_1st_Layer5, OutImage_1st_Layer5, OutImage_1st_Layer7, OutImage_1st_Layer8, Output_Layer8, Bias_Grad
        # Layer 0
        # Read DDR & Conver Format # 512MB
        layer0_start = time.time()

        s = time.time()
        Layer0_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x83E00000, End_Address=0x83ED0000)
        Layer0_1st_Iter_Image1_CH0_256 = data_32_to_16(Layer0_1st_Iter_Image1_CH0) 
        #if DEBUG: print("ch0 image 1 : ", len(Layer0_1st_Iter_Image1_CH0)) 

        Layer0_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x83ED0000, End_Address=0x83FA0000)
        Layer0_1st_Iter_Image2_CH0_256 = data_32_to_16(Layer0_1st_Iter_Image2_CH0)
        #if DEBUG: print("ch0 image 2 : ", len(Layer0_1st_Iter_Image2_CH0))
        
        Layer0_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x83FA0000, End_Address=0x84070000)
        Layer0_1st_Iter_Image3_CH0_256 = data_32_to_16(Layer0_1st_Iter_Image3_CH0)
        #if DEBUG: print("ch0 image 3 : ", len(Layer0_1st_Iter_Image3_CH0))

        Layer0_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x84070000, End_Address=0x84140000)
        Layer0_1st_Iter_Image4_CH0_256 = data_32_to_16(Layer0_1st_Iter_Image4_CH0)
        #if DEBUG: print("ch0 image 4 : ", len(Layer0_1st_Iter_Image4_CH0))

        Layer0_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x84140000, End_Address=0x84210000)
        Layer0_1st_Iter_Image5_CH0_256 = data_32_to_16(Layer0_1st_Iter_Image5_CH0)
        #if DEBUG: print("ch0 image 5 : ", len(Layer0_1st_Iter_Image5_CH0))

        Layer0_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x84210000, End_Address=0x842E0000)
        Layer0_1st_Iter_Image6_CH0_256 = data_32_to_16(Layer0_1st_Iter_Image6_CH0)
        #if DEBUG: print("ch0 image 6 : ", len(Layer0_1st_Iter_Image6_CH0))

        Layer0_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x842E0000, End_Address=0x843B0000)
        Layer0_1st_Iter_Image7_CH0_256 = data_32_to_16(Layer0_1st_Iter_Image7_CH0)
        #if DEBUG: print("ch0 image 7 : ", len(Layer0_1st_Iter_Image7_CH0))

        Layer0_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x843B0000, End_Address=0x84480000)
        Layer0_1st_Iter_Image8_CH0_256 = data_32_to_16(Layer0_1st_Iter_Image8_CH0)
        #if DEBUG: print("ch0 image 8 : ", len(Layer0_1st_Iter_Image8_CH0))


        Layer0_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x93E00000, End_Address=0x93ED0000)
        Layer0_1st_Iter_Image1_CH1_256 = data_32_to_16(Layer0_1st_Iter_Image1_CH1)
        #if DEBUG: print("ch1 image 1 : ", len(Layer0_1st_Iter_Image1_CH1))

        Layer0_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x93ED0000, End_Address=0x93FA0000)
        Layer0_1st_Iter_Image2_CH1_256 = data_32_to_16(Layer0_1st_Iter_Image2_CH1)
        #if DEBUG: print("ch1 image 2 : ", len(Layer0_1st_Iter_Image2_CH1))

        Layer0_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x93FA0000, End_Address=0x94070000)
        Layer0_1st_Iter_Image3_CH1_256 = data_32_to_16(Layer0_1st_Iter_Image3_CH1)
        #if DEBUG: print("ch1 image 3 : ", len(Layer0_1st_Iter_Image3_CH1))

        Layer0_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x94070000, End_Address=0x94140000)
        Layer0_1st_Iter_Image4_CH1_256 = data_32_to_16(Layer0_1st_Iter_Image4_CH1)
        #if DEBUG: print("ch1 image 4 : ", len(Layer0_1st_Iter_Image4_CH1))

        Layer0_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x94140000, End_Address=0x94210000)
        Layer0_1st_Iter_Image5_CH1_256 = data_32_to_16(Layer0_1st_Iter_Image5_CH1)
        #if DEBUG: print("ch1 image 5 : ", len(Layer0_1st_Iter_Image5_CH1))

        Layer0_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x94210000, End_Address=0x942E0000)
        Layer0_1st_Iter_Image6_CH1_256 = data_32_to_16(Layer0_1st_Iter_Image6_CH1)
        #if DEBUG: print("ch1 image 6 : ", len(Layer0_1st_Iter_Image6_CH1))

        Layer0_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x942E0000, End_Address=0x943B0000)
        Layer0_1st_Iter_Image7_CH1_256 = data_32_to_16(Layer0_1st_Iter_Image7_CH1)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Layer0_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x943B0000, End_Address=0x94480000)
        Layer0_1st_Iter_Image8_CH1_256 = data_32_to_16(Layer0_1st_Iter_Image8_CH1)
        #if DEBUG: print("ch1 image 8 : ", len(Layer0_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR & 32bit to 16bit Convert :",e-s)

        '''
        test_out = '1st_iter_result/Layer0_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image1_CH0:
                test_output.write(str(item) + "\n")
        test_output.close()
        
        test_out = '1st_iter_result/Layer0_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''

        s = time.time()
        Output_Image1_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image1_CH0_256, Layer0_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image2_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image2_CH0_256, Layer0_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image3_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image3_CH0_256, Layer0_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image4_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image4_CH0_256, Layer0_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image5_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image5_CH0_256, Layer0_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image6_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image6_CH0_256, Layer0_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image7_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image7_CH0_256, Layer0_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image8_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image8_CH0_256, Layer0_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)

        OutImages_1st_Layer0 = Output_Image1_Layer0_1st_Iter + Output_Image2_Layer0_1st_Iter + Output_Image3_Layer0_1st_Iter + Output_Image4_Layer0_1st_Iter + \
                            Output_Image5_Layer0_1st_Iter + Output_Image6_Layer0_1st_Iter + Output_Image7_Layer0_1st_Iter + Output_Image8_Layer0_1st_Iter    

        OutImage_1st_Layer0 = torch.tensor([float(value) for value in OutImages_1st_Layer0], dtype=torch.float32).reshape(8, 16, 208, 208)
        
        if DEBUG2 : Save_File(OutImage_1st_Layer0, "result/Layer_0_Forward_1st_Iteration")
        
        if DEBUG: print(OutImage_1st_Layer0[0][0][0][0:5])
        
        # Mean, Var
        s = time.time()
        Mean_1st_Layer0, Var_1st_Layer0 = Cal_mean_var.forward(OutImage_1st_Layer0)    
        e = time.time()
        if DEBUG: print("Calculate Mean & Var :",e-s)

        Beta_Layer0 = data.Beta_Dec[0]
        Gamma_Layer0 = data.Gamma_Dec[0]

        if DEBUG2 : Save_File(data.Beta_Dec[0],   "result/Layer_0_Forward_Beta_Before_Weight_Update")
        if DEBUG2 : Save_File(data.Gamma_Dec[0],  "result/Layer_0_Forward_Gamma_Before_Weight_Update")
        if DEBUG2 : Save_File(data.Weight_Dec[0], "result/Layer_0_Forward_weight_Before_Weight_Update")

        # layer0 Caches: 
        layer0_cache = BN(OutImage_1st_Layer0, Gamma_Layer0, Beta_Layer0)

        # Squeeze to remove the dimension but keeping the same data ordering
        Var_1st_Layer0 = Var_1st_Layer0.squeeze() * Gamma_Layer0
        s = time.time()
        Mean_1st_Layer0, Var_1st_Layer0 = Mean_Var_Dec2Bfloat(Mean_1st_Layer0, Var_1st_Layer0, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat :",e-s)
        s= time.time()
        Weight_2nd_Layer0 = New_Weight_Hardware_ReOrdering_Layer0(16, 16, data.Weight_Bfloat[0], Mean_1st_Layer0, Var_1st_Layer0, data.Beta_Bfloat[0], Iteration="2")
        #if DEBUG: print("Weight_2nd_Layer0 : ", Weight_2nd_Layer0)
        e = time.time()
 
        '''
        data_read_mean_var = "result/Mean_1st_Layer0.txt"
        with open(data_read_mean_var, mode="w") as output_file:  
            for sublist in Mean_1st_Layer0:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n") 
        output_file.close() 

        data_read_mean_var = "result/Var_1st_Layer0.txt"
        with open(data_read_mean_var, mode="w") as output_file:  
            for sublist in Var_1st_Layer0:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n") 
        output_file.close() 
        
        data_read_mean_var = "result/layer0_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:  
            for sublist in Weight_2nd_Layer0:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n") 
        output_file.close()               
        '''
 
        '''
        weight_layer0_2nd_ch0 = data_256_32(Weight_2nd_Layer0[0])
        weight_layer0_2nd_ch1 = data_256_32(Weight_2nd_Layer0[1])
        
        data_read_mean_var = "result/weight_layer0_2nd_ch0.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in weight_layer0_2nd_ch0:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")      
        output_file.close()  

        data_read_mean_var = "result/weight_layer0_2nd_ch1.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in weight_layer0_2nd_ch1:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write
        '''
        s = time.time()
        Write_DDR(data_256_32(Weight_2nd_Layer0[0]), Wr_Address=0x80000000)
        Write_DDR(data_256_32(Weight_2nd_Layer0[1]), Wr_Address=0x90000000)
        e = time.time()
        if DEBUG: print("Write DDR & 256bit to 32bit",e-s)
        
        resume()
        layer0_end = time.time()
        process = layer0_end - layer0_start
        if DEBUG: print("layer0 process time : ", process)

        '''
        d = Device("0000:08:00.0")
        bar = d.bar[0]

        data_read = open("result/layer0_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 
        
        d = Device("0000:08:00.0")
        bar = d.bar[2]
        
        data_read = open("result/layer0_result_ch0_image1.txt", mode="w+")
        i=0
        for i in range(0,int((0X4550000-0X4480000)/4) ): 
            Read_Data = bar.read(0X4480000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer0_result_ch1_image1.txt", mode="w+")
        i=0
        for i in range(0,int((0X4550000-0X4480000)/4) ): 
            Read_Data = bar.read(0X14480000 + (i*4))
            data_read.write(str(Read_Data) + "\n")     

        data_read = open("result/layer0_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X784C000-0X71CC000)/4) ): 
            Read_Data = bar.read(0X71CC000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      
        '''
            
        #################################################
        #                Layer 1 Start                  #
        #################################################
        # check Layer1 IRQ
        check_irq_otherlayer()     
        # self.app_instance .change_color(self.app_instance.L2_IRQ_canvas, self.app_instance.L2_IRQ, "green") 
        # Layer 1
        
        Image0_2nd_ch0 = Read_DDR(Rd_Address=0X84480000, End_Address=0X84550000)
        Image0_2nd_ch0 = data_32_to_16(Image0_2nd_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Image0_2nd_ch1 = Read_DDR(Rd_Address=0X94480000, End_Address=0X94550000)
        Image0_2nd_ch1 = data_32_to_16(Image0_2nd_ch1)
        
        Image0_2nd_result = Read_OutFmap_Bfloat2Dec(Image0_2nd_ch0, Image0_2nd_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False) 

        Image1_2nd_ch0 = Read_DDR(Rd_Address=0X84550000, End_Address=0X84620000)
        Image1_2nd_ch0 = data_32_to_16(Image1_2nd_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Image1_2nd_ch1 = Read_DDR(Rd_Address=0X94550000, End_Address=0X94620000)
        Image1_2nd_ch1 = data_32_to_16(Image1_2nd_ch1)
        
        Image1_2nd_result = Read_OutFmap_Bfloat2Dec(Image1_2nd_ch0, Image1_2nd_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False) 

        Image2_2nd_ch0 = Read_DDR(Rd_Address=0X84620000, End_Address=0X846F0000)
        Image2_2nd_ch0 = data_32_to_16(Image2_2nd_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Image2_2nd_ch1 = Read_DDR(Rd_Address=0X94620000, End_Address=0X946F0000)
        Image2_2nd_ch1 = data_32_to_16(Image2_2nd_ch1)
        
        Image2_2nd_result = Read_OutFmap_Bfloat2Dec(Image2_2nd_ch0, Image2_2nd_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False) 

        Image3_2nd_ch0 = Read_DDR(Rd_Address=0X846F0000, End_Address=0X847C0000)
        Image3_2nd_ch0 = data_32_to_16(Image3_2nd_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Image3_2nd_ch1 = Read_DDR(Rd_Address=0X946F0000, End_Address=0X947C0000)
        Image3_2nd_ch1 = data_32_to_16(Image3_2nd_ch1)
        
        Image3_2nd_result = Read_OutFmap_Bfloat2Dec(Image3_2nd_ch0, Image3_2nd_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False) 

        Image4_2nd_ch0 = Read_DDR(Rd_Address=0X847C0000, End_Address=0X84890000)
        Image4_2nd_ch0 = data_32_to_16(Image4_2nd_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Image4_2nd_ch1 = Read_DDR(Rd_Address=0X947C0000, End_Address=0X94890000)
        Image4_2nd_ch1 = data_32_to_16(Image4_2nd_ch1)
        
        Image4_2nd_result = Read_OutFmap_Bfloat2Dec(Image4_2nd_ch0, Image4_2nd_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False) 
        
        Image5_2nd_ch0 = Read_DDR(Rd_Address=0X84890000, End_Address=0X84960000)
        Image5_2nd_ch0 = data_32_to_16(Image5_2nd_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Image5_2nd_ch1 = Read_DDR(Rd_Address=0X94890000, End_Address=0X94960000)
        Image5_2nd_ch1 = data_32_to_16(Image5_2nd_ch1)
        
        Image5_2nd_result = Read_OutFmap_Bfloat2Dec(Image5_2nd_ch0, Image5_2nd_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False) 

        Image6_2nd_ch0 = Read_DDR(Rd_Address=0X84960000, End_Address=0X84A30000)
        Image6_2nd_ch0 = data_32_to_16(Image6_2nd_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Image6_2nd_ch1 = Read_DDR(Rd_Address=0X94960000, End_Address=0X94A30000)
        Image6_2nd_ch1 = data_32_to_16(Image6_2nd_ch1)
        
        Image6_2nd_result = Read_OutFmap_Bfloat2Dec(Image6_2nd_ch0, Image6_2nd_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False) 

        Image7_2nd_ch0 = Read_DDR(Rd_Address=0X84A30000, End_Address=0X84B00000)
        Image7_2nd_ch0 = data_32_to_16(Image7_2nd_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Image7_2nd_ch1 = Read_DDR(Rd_Address=0X94A30000, End_Address=0X94B00000)
        Image7_2nd_ch1 = data_32_to_16(Image7_2nd_ch1)
        
        Image7_2nd_result = Read_OutFmap_Bfloat2Dec(Image7_2nd_ch0, Image7_2nd_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False) 

        Image_2nd_result = Image0_2nd_result + Image1_2nd_result + Image2_2nd_result + Image3_2nd_result + Image4_2nd_result + Image5_2nd_result +\
                           Image6_2nd_result + Image7_2nd_result
        
        Image_2nd_result = torch.tensor([float(value) for value in Image_2nd_result], dtype=torch.float32).reshape(8, 16, 208, 208)
        
        # Save_File(Image0_2nd_result, "result/Image0_2nd_result")
        # Save_File(Image1_2nd_result, "result/Image1_2nd_result")
        # Save_File(Image2_2nd_result, "result/Image2_2nd_result")
        # Save_File(Image3_2nd_result, "result/Image3_2nd_result")
        # Save_File(Image4_2nd_result, "result/Image4_2nd_result")
        # Save_File(Image5_2nd_result, "result/Image5_2nd_result")
        # Save_File(Image6_2nd_result, "result/Image6_2nd_result")
        # Save_File(Image7_2nd_result, "result/Image7_2nd_result")
        # Save_File(Image_2nd_result, "result/Image_2nd_result")
        
        _Out_2nd_Itr = [Image0_2nd_result, Image1_2nd_result, Image2_2nd_result, Image3_2nd_result, Image4_2nd_result, Image5_2nd_result, Image6_2nd_result, Image7_2nd_result]
        if DEBUG2 : Save_File(_Out_2nd_Itr, "result/Layer_0_Forward_2nd_Iteration") 
        
        
        
        layer1_start = time.time()
        s = time.time()
        # Read DDR & Conver Format # 512MB
        Layer1_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x84B00000, End_Address=0x84CA0000)
        Layer1_1st_Iter_Image1_CH0_256 = (data_32_to_16(Layer1_1st_Iter_Image1_CH0))   
        #if DEBUG: print("ch0 image 1 : ", len(Layer1_1st_Iter_Image1_CH0))    
        
        Layer1_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x84CA0000, End_Address=0x84E40000)
        Layer1_1st_Iter_Image2_CH0_256 = (data_32_to_16(Layer1_1st_Iter_Image2_CH0))
        #if DEBUG: print("ch0 image 2 : ", len(Layer1_1st_Iter_Image2_CH0))
        
        Layer1_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x84E40000, End_Address=0x84FE0000)
        Layer1_1st_Iter_Image3_CH0_256 = (data_32_to_16(Layer1_1st_Iter_Image3_CH0))
        #if DEBUG: print("ch0 image 3 : ", len(Layer1_1st_Iter_Image3_CH0))

        Layer1_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x84FE0000, End_Address=0x85180000)
        Layer1_1st_Iter_Image4_CH0_256 = (data_32_to_16(Layer1_1st_Iter_Image4_CH0))
        #if DEBUG: print("ch0 image 4 : ", len(Layer1_1st_Iter_Image4_CH0))

        Layer1_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x85180000, End_Address=0x85320000)
        Layer1_1st_Iter_Image5_CH0_256 = (data_32_to_16(Layer1_1st_Iter_Image5_CH0))
        #if DEBUG: print("ch0 image 5 : ", len(Layer1_1st_Iter_Image5_CH0))

        Layer1_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x85320000, End_Address=0x854C0000)
        Layer1_1st_Iter_Image6_CH0_256 = (data_32_to_16(Layer1_1st_Iter_Image6_CH0))
        #if DEBUG: print("ch0 image 6 : ", len(Layer1_1st_Iter_Image6_CH0))

        Layer1_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x854C0000, End_Address=0x85660000)
        Layer1_1st_Iter_Image7_CH0_256 = (data_32_to_16(Layer1_1st_Iter_Image7_CH0))
        #if DEBUG: print("ch0 image 7 : ", len(Layer1_1st_Iter_Image7_CH0))

        Layer1_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x85660000, End_Address=0x85800000)
        Layer1_1st_Iter_Image8_CH0_256 = (data_32_to_16(Layer1_1st_Iter_Image8_CH0))
        #if DEBUG: print("ch0 image 8 : ", len(Layer1_1st_Iter_Image8_CH0))


        Layer1_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x94B00000, End_Address=0x94CA0000)
        Layer1_1st_Iter_Image1_CH1_256 = (data_32_to_16(Layer1_1st_Iter_Image1_CH1))
        #if DEBUG: print("ch1 image 1 : ", len(Layer1_1st_Iter_Image1_CH1))

        Layer1_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x94CA0000, End_Address=0x94E40000)
        Layer1_1st_Iter_Image2_CH1_256 = (data_32_to_16(Layer1_1st_Iter_Image2_CH1))
        #if DEBUG: print("ch1 image 2 : ", len(Layer1_1st_Iter_Image2_CH1))

        Layer1_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x94E40000, End_Address=0x94FE0000)
        Layer1_1st_Iter_Image3_CH1_256 = (data_32_to_16(Layer1_1st_Iter_Image3_CH1))
        #if DEBUG: print("ch1 image 3 : ", len(Layer1_1st_Iter_Image3_CH1))

        Layer1_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x94FE0000, End_Address=0x95180000)
        Layer1_1st_Iter_Image4_CH1_256 = (data_32_to_16(Layer1_1st_Iter_Image4_CH1))
        #if DEBUG: print("ch1 image 4 : ", len(Layer1_1st_Iter_Image4_CH1))

        Layer1_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x95180000, End_Address=0x95320000)
        Layer1_1st_Iter_Image5_CH1_256 = (data_32_to_16(Layer1_1st_Iter_Image5_CH1))
        #if DEBUG: print("ch1 image 5 : ", len(Layer1_1st_Iter_Image5_CH1))

        Layer1_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x95320000, End_Address=0x954C0000)
        Layer1_1st_Iter_Image6_CH1_256 = (data_32_to_16(Layer1_1st_Iter_Image6_CH1))
        #if DEBUG: print("ch1 image 6 : ", len(Layer1_1st_Iter_Image6_CH1))

        Layer1_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x954C0000, End_Address=0x95660000)
        Layer1_1st_Iter_Image7_CH1_256 = (data_32_to_16(Layer1_1st_Iter_Image7_CH1))
        #if DEBUG: print("ch1 image 7 : ", len(Layer1_1st_Iter_Image7_CH1))

        Layer1_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x95660000, End_Address=0x95800000)
        Layer1_1st_Iter_Image8_CH1_256 = (data_32_to_16(Layer1_1st_Iter_Image8_CH1))
        #if DEBUG: print("ch1 image 8 : ", len(Layer1_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR & 32bit to 16bit :",e-s)

        '''
        test_out = '1st_iter_result/Layer1_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''

        s = time.time()
        Output_Image1_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image1_CH0_256, Layer1_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image2_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image2_CH0_256, Layer1_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image3_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image3_CH0_256, Layer1_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image4_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image4_CH0_256, Layer1_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image5_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image5_CH0_256, Layer1_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image6_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image6_CH0_256, Layer1_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image7_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image7_CH0_256, Layer1_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image8_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image8_CH0_256, Layer1_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec :",e-s)
        
        OutImages_1st_Layer1 = Output_Image1_Layer1_1st_Iter + Output_Image2_Layer1_1st_Iter + Output_Image3_Layer1_1st_Iter + Output_Image4_Layer1_1st_Iter + \
                            Output_Image5_Layer1_1st_Iter + Output_Image6_Layer1_1st_Iter + Output_Image7_Layer1_1st_Iter + Output_Image8_Layer1_1st_Iter    

        OutImage_1st_Layer1 = torch.tensor([float(value) for value in OutImages_1st_Layer1], dtype=torch.float32).reshape(8, 32, 208, 208)
        if DEBUG: print(OutImage_1st_Layer1[0][0][0][0:5])
        
        if DEBUG2 : Save_File(data.Beta_Dec[1],   "result/Layer_1_Forward_Beta_Before_Weight_Update")
        if DEBUG2 : Save_File(data.Gamma_Dec[1],  "result/Layer_1_Forward_Gamma_Before_Weight_Update")
        if DEBUG2 : Save_File(data.Weight_Dec[1], "result/Layer_1_Forward_weight_Before_Weight_Update")
        
        # Mean, Var
        s = time.time()
        Mean_1st_Layer1, Var_1st_Layer1 = Cal_mean_var.forward(OutImage_1st_Layer1)
        e = time.time()
        if DEBUG: print("Cacluate Mean & Var :",e-s)
        
        Beta_Layer1 = data.Beta_Dec[1]
        Gamma_Layer1 = data.Gamma_Dec[1]

        layer1_cache = BN(OutImage_1st_Layer1, Gamma_Layer1, Beta_Layer1)

        # Squeeze to remove the dimension but keeping the same data ordering
        Var_1st_Layer1 = Var_1st_Layer1.squeeze() * Gamma_Layer1

        s = time.time()
        Mean_1st_Layer1, Var_1st_Layer1 = Mean_Var_Dec2Bfloat(Mean_1st_Layer1, Var_1st_Layer1, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat :",e-s)
        s = time.time()
        Weight_2nd_Layer1 = New_Weight_Hardware_ReOrdering_OtherLayer(32, 16, data.Weight_Bfloat[1], Mean_1st_Layer1, Var_1st_Layer1, data.Beta_Bfloat[1], Iteration="2")
        e = time.time()
        if DEBUG: print("Weight Reordering :",e-s)

        # Write DDR
        s = time.time()
        Write_DDR(data_256_32(Weight_2nd_Layer1[0]), Wr_Address=0x80000A00)
        Write_DDR(data_256_32(Weight_2nd_Layer1[1]), Wr_Address=0x90000A00)
        e = time.time()
        if DEBUG: print("Write DDR & 256bit to 32 bit :",e-s)

        layer1_end = time.time()
        layer1_process = layer1_end - layer1_start
        if DEBUG: print("Layer1 process time : ", layer1_process)

        resume()

        '''
        d = Device("0000:08:00.0")
        bar = d.bar[0]

        data_read = open("result/layer1_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        d = Device("0000:08:00.0")
        bar = d.bar[2]

        data_read = open("result/layer1_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X5B40000-0X5800000)/4) ): 
            Read_Data = bar.read(0X5800000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer1_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X5B40000-0X5800000)/4) ): 
            Read_Data = bar.read(0X15800000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer1_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X7B8C000-0X784C000)/4) ): 
            Read_Data = bar.read(0X784C000 + (i*4))
            data_read.write(str(Read_Data) + "\n")     
        '''   

        #################################################
        #                Layer 2 Start                  #
        #################################################
        # check Layer2 IRQ
        check_irq_otherlayer()
        # self.app_instance .change_color(self.app_instance.L3_IRQ_canvas, self.app_instance.L3_IRQ, "green")
        # Layer 2
        layer2_start = time.time()
        # Read DDR & Conver Format # 512MB
        s = time.time()
        Layer2_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x85B40000, End_Address=0x85C10000)
        Layer2_1st_Iter_Image1_CH0_256 = (data_32_to_16(Layer2_1st_Iter_Image1_CH0))   
        #if DEBUG: print("ch0 image 1 : ", len(Layer2_1st_Iter_Image1_CH0))     

        Layer2_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x85C10000, End_Address=0x85CE0000)
        Layer2_1st_Iter_Image2_CH0_256 = (data_32_to_16(Layer2_1st_Iter_Image2_CH0))
        #if DEBUG: print("ch0 image 2 : ", len(Layer2_1st_Iter_Image2_CH0))
        
        Layer2_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x85CE0000, End_Address=0x85DB0000)
        Layer2_1st_Iter_Image3_CH0_256 = (data_32_to_16(Layer2_1st_Iter_Image3_CH0))
        #if DEBUG: print("ch0 image 3 : ", len(Layer2_1st_Iter_Image3_CH0))

        Layer2_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x85DB0000, End_Address=0x85E80000)
        Layer2_1st_Iter_Image4_CH0_256 = (data_32_to_16(Layer2_1st_Iter_Image4_CH0))
        #if DEBUG: print("ch0 image 4 : ", len(Layer2_1st_Iter_Image4_CH0))

        Layer2_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x85E80000, End_Address=0x85F50000)
        Layer2_1st_Iter_Image5_CH0_256 = (data_32_to_16(Layer2_1st_Iter_Image5_CH0))
        #if DEBUG: print("ch0 image 5 : ", len(Layer2_1st_Iter_Image5_CH0))

        Layer2_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x85F50000, End_Address=0x86020000)
        Layer2_1st_Iter_Image6_CH0_256 = (data_32_to_16(Layer2_1st_Iter_Image6_CH0))
        #if DEBUG: print("ch0 image 6 : ", len(Layer2_1st_Iter_Image6_CH0))

        Layer2_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x86020000, End_Address=0x860F0000)
        Layer2_1st_Iter_Image7_CH0_256 = (data_32_to_16(Layer2_1st_Iter_Image7_CH0))
        #if DEBUG: print("ch0 image 7 : ", len(Layer2_1st_Iter_Image7_CH0))

        Layer2_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x860F0000, End_Address=0x861C0000)
        Layer2_1st_Iter_Image8_CH0_256 = (data_32_to_16(Layer2_1st_Iter_Image8_CH0))
        #if DEBUG: print("ch0 image 8 : ", len(Layer2_1st_Iter_Image8_CH0))


        Layer2_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x95B40000, End_Address=0x95C10000)
        Layer2_1st_Iter_Image1_CH1_256 = (data_32_to_16(Layer2_1st_Iter_Image1_CH1))
        #if DEBUG: print("ch1 image 1 : ", len(Layer2_1st_Iter_Image1_CH1))

        Layer2_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x95C10000, End_Address=0x95CE0000)
        Layer2_1st_Iter_Image2_CH1_256 = (data_32_to_16(Layer2_1st_Iter_Image2_CH1))
        #if DEBUG: print("ch1 image 2 : ", len(Layer2_1st_Iter_Image2_CH1))

        Layer2_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x95CE0000, End_Address=0x95DB0000)
        Layer2_1st_Iter_Image3_CH1_256 = (data_32_to_16(Layer2_1st_Iter_Image3_CH1))
        #if DEBUG: print("ch1 image 3 : ", len(Layer2_1st_Iter_Image3_CH1))

        Layer2_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x95DB0000, End_Address=0x95E80000)
        Layer2_1st_Iter_Image4_CH1_256 = (data_32_to_16(Layer2_1st_Iter_Image4_CH1))
        #if DEBUG: print("ch1 image 4 : ", len(Layer2_1st_Iter_Image4_CH1))

        Layer2_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x95E80000, End_Address=0x95F50000)
        Layer2_1st_Iter_Image5_CH1_256 = (data_32_to_16(Layer2_1st_Iter_Image5_CH1))
        #if DEBUG: print("ch1 image 5 : ", len(Layer2_1st_Iter_Image5_CH1))

        Layer2_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x95F50000, End_Address=0x96020000)
        Layer2_1st_Iter_Image6_CH1_256 = (data_32_to_16(Layer2_1st_Iter_Image6_CH1))
        #if DEBUG: print("ch1 image 6 : ", len(Layer2_1st_Iter_Image6_CH1))

        Layer2_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x96020000, End_Address=0x960F0000)
        Layer2_1st_Iter_Image7_CH1_256 = (data_32_to_16(Layer2_1st_Iter_Image7_CH1))
        #if DEBUG: print("ch1 image 7 : ", len(Layer2_1st_Iter_Image7_CH1))

        Layer2_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x960F0000, End_Address=0x961C0000)
        Layer2_1st_Iter_Image8_CH1_256 = (data_32_to_16(Layer2_1st_Iter_Image8_CH1))
        #if DEBUG: print("ch1 image 8 : ", len(Layer2_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR & 32bit to 16bit :",e-s)

        '''
        test_out = '1st_iter_result/Layer2_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''

        s = time.time()
        Output_Image1_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image1_CH0_256, Layer2_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image2_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image2_CH0_256, Layer2_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image3_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image3_CH0_256, Layer2_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image4_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image4_CH0_256, Layer2_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image5_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image5_CH0_256, Layer2_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image6_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image6_CH0_256, Layer2_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image7_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image7_CH0_256, Layer2_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image8_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image8_CH0_256, Layer2_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec :",e-s)

        OutImages_1st_Layer2 = Output_Image1_Layer2_1st_Iter + Output_Image2_Layer2_1st_Iter + Output_Image3_Layer2_1st_Iter + Output_Image4_Layer2_1st_Iter + \
                            Output_Image5_Layer2_1st_Iter + Output_Image6_Layer2_1st_Iter + Output_Image7_Layer2_1st_Iter + Output_Image8_Layer2_1st_Iter    

        OutImage_1st_Layer2 = torch.tensor([float(value) for value in OutImages_1st_Layer2], dtype=torch.float32).reshape(8, 64, 104, 104)
        # Mean, Var
        s = time.time()
        Mean_1st_Layer2, Var_1st_Layer2 = Cal_mean_var.forward(OutImage_1st_Layer2)
        e = time.time()
        if DEBUG: print("Calcuulate Mean & Var :",e-s)

        Beta_Layer2 = data.Beta_Dec[2]
        Gamma_Layer2 = data.Gamma_Dec[2]

        layer2_cache = BN(OutImage_1st_Layer2, Gamma_Layer2, Beta_Layer2)

        # Squeeze to remove the dimension but keeping the same data ordering
        Var_1st_Layer2 = Var_1st_Layer2.squeeze() * Gamma_Layer2

        s = time.time()
        Mean_1st_Layer2, Var_1st_Layer2 = Mean_Var_Dec2Bfloat(Mean_1st_Layer2, Var_1st_Layer2, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat :",e-s)
        s = time.time()
        Weight_2nd_Layer2 = New_Weight_Hardware_ReOrdering_OtherLayer(64, 32, data.Weight_Bfloat[2], Mean_1st_Layer2, Var_1st_Layer2, data.Beta_Bfloat[2], Iteration="2")
        e = time.time()
        if DEBUG: print("Weight Reordering :",e-s)

        '''
        data_read_mean_var = "result/layer2_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer2:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")     
        output_file.close()    
        '''       

        # Write DDR
        s = time.time()
        Write_DDR(data_256_32(Weight_2nd_Layer2[0]), Wr_Address=0x80001E00)
        Write_DDR(data_256_32(Weight_2nd_Layer2[1]), Wr_Address=0x90001E00)
        e = time.time()
        if DEBUG: print("Write DDR & 256bit to 32bit :",e-s)
        
        layer2_end = time.time()
        layer2_process = layer2_end - layer2_start
        if DEBUG: print("Layer2 process time : ", layer2_process)

        resume()

        '''
        d = Device("0000:08:00.0")
        bar = d.bar[0]

        data_read = open("result/layer_2_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        d = Device("0000:08:00.0")
        bar = d.bar[2]

        data_read = open("result/layer2_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X6360000-0X61C0000)/4) ): 
            Read_Data = bar.read(0X61C0000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer2_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X6360000-0X61C0000)/4) ): 
            Read_Data = bar.read(0X161C0000 + (i*4))
            data_read.write(str(Read_Data) + "\n")

        data_read = open("result/layer2_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X7D2C000-0X7B8C000)/4) ): 
            Read_Data = bar.read(0X7B8C000 + (i*4))
            data_read.write(str(Read_Data) + "\n")     
        '''


        #################################################
        #                Layer 3 Start                  #
        #################################################
        # check Layer3 IRQ
        check_irq_otherlayer()
        # self.app_instance .change_color(self.app_instance.L4_IRQ_canvas, self.app_instance.L4_IRQ, "green")
        # Layer 3
        layer3_start = time.time()
        # Read DDR & Conver Format # 512MB
        s = time.time()
        Layer3_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x86360000, End_Address=0x863C8000)
        Layer3_1st_Iter_Image1_CH0_256 = (data_32_to_16(Layer3_1st_Iter_Image1_CH0))   
        #if DEBUG: print("ch0 image 1 : ", len(Layer3_1st_Iter_Image1_CH0))     

        Layer3_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x863C8000, End_Address=0x86430000)
        Layer3_1st_Iter_Image2_CH0_256 = (data_32_to_16(Layer3_1st_Iter_Image2_CH0))
        #if DEBUG: print("ch0 image 2 : ", len(Layer3_1st_Iter_Image2_CH0))
        
        Layer3_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x86430000, End_Address=0x86498000)
        Layer3_1st_Iter_Image3_CH0_256 = (data_32_to_16(Layer3_1st_Iter_Image3_CH0))
        #if DEBUG: print("ch0 image 3 : ", len(Layer3_1st_Iter_Image3_CH0))

        Layer3_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x86498000, End_Address=0x86500000)
        Layer3_1st_Iter_Image4_CH0_256 = (data_32_to_16(Layer3_1st_Iter_Image4_CH0))
        #if DEBUG: print("ch0 image 4 : ", len(Layer3_1st_Iter_Image4_CH0))

        Layer3_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x86500000, End_Address=0x86568000)
        Layer3_1st_Iter_Image5_CH0_256 = (data_32_to_16(Layer3_1st_Iter_Image5_CH0))
        #if DEBUG: print("ch0 image 5 : ", len(Layer3_1st_Iter_Image5_CH0))

        Layer3_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x86568000, End_Address=0x865D0000)
        Layer3_1st_Iter_Image6_CH0_256 = (data_32_to_16(Layer3_1st_Iter_Image6_CH0))
        #if DEBUG: print("ch0 image 6 : ", len(Layer3_1st_Iter_Image6_CH0))

        Layer3_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x865D0000, End_Address=0x86638000)
        Layer3_1st_Iter_Image7_CH0_256 = (data_32_to_16(Layer3_1st_Iter_Image7_CH0))
        #if DEBUG: print("ch0 image 7 : ", len(Layer3_1st_Iter_Image7_CH0))

        Layer3_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x86638000, End_Address=0x866A0000)
        Layer3_1st_Iter_Image8_CH0_256 = (data_32_to_16(Layer3_1st_Iter_Image8_CH0))
        #if DEBUG: print("ch0 image 8 : ", len(Layer3_1st_Iter_Image8_CH0))


        Layer3_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x96360000, End_Address=0x963C8000)
        Layer3_1st_Iter_Image1_CH1_256 = (data_32_to_16(Layer3_1st_Iter_Image1_CH1))
        #if DEBUG: print("ch1 image 1 : ", len(Layer3_1st_Iter_Image1_CH1))

        Layer3_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x963C8000, End_Address=0x96430000)
        Layer3_1st_Iter_Image2_CH1_256 = (data_32_to_16(Layer3_1st_Iter_Image2_CH1))
        #if DEBUG: print("ch1 image 2 : ", len(Layer3_1st_Iter_Image2_CH1))

        Layer3_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x96430000, End_Address=0x96498000)
        Layer3_1st_Iter_Image3_CH1_256 = (data_32_to_16(Layer3_1st_Iter_Image3_CH1))
        #if DEBUG: print("ch1 image 3 : ", len(Layer3_1st_Iter_Image3_CH1))

        Layer3_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x96498000, End_Address=0x96500000)
        Layer3_1st_Iter_Image4_CH1_256 = (data_32_to_16(Layer3_1st_Iter_Image4_CH1))
        #if DEBUG: print("ch1 image 4 : ", len(Layer3_1st_Iter_Image4_CH1))

        Layer3_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x96500000, End_Address=0x96568000)
        Layer3_1st_Iter_Image5_CH1_256 = (data_32_to_16(Layer3_1st_Iter_Image5_CH1))
        #if DEBUG: print("ch1 image 5 : ", len(Layer3_1st_Iter_Image5_CH1))

        Layer3_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x96568000, End_Address=0x965D0000)
        Layer3_1st_Iter_Image6_CH1_256 = (data_32_to_16(Layer3_1st_Iter_Image6_CH1))
        #if DEBUG: print("ch1 image 6 : ", len(Layer3_1st_Iter_Image6_CH1))

        Layer3_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x965D0000, End_Address=0x96638000)
        Layer3_1st_Iter_Image7_CH1_256 = (data_32_to_16(Layer3_1st_Iter_Image7_CH1))
        #if DEBUG: print("ch1 image 7 : ", len(Layer3_1st_Iter_Image7_CH1))

        Layer3_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x96638000, End_Address=0x966A0000)
        Layer3_1st_Iter_Image8_CH1_256 = (data_32_to_16(Layer3_1st_Iter_Image8_CH1))
        #if DEBUG: print("ch1 image 8 : ", len(Layer3_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR & 32bit to 16bit : ",e-s)

        '''
        test_out = '1st_iter_result/Layer3_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''
        
        s = time.time()
        Output_Image1_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image1_CH0_256, Layer3_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image2_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image2_CH0_256, Layer3_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image3_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image3_CH0_256, Layer3_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image4_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image4_CH0_256, Layer3_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image5_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image5_CH0_256, Layer3_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image6_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image6_CH0_256, Layer3_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image7_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image7_CH0_256, Layer3_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image8_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image8_CH0_256, Layer3_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)

        OutImages_1st_Layer3 = Output_Image1_Layer3_1st_Iter + Output_Image2_Layer3_1st_Iter + Output_Image3_Layer3_1st_Iter + Output_Image4_Layer3_1st_Iter + \
                            Output_Image5_Layer3_1st_Iter + Output_Image6_Layer3_1st_Iter + Output_Image7_Layer3_1st_Iter + Output_Image8_Layer3_1st_Iter    

        OutImage_1st_Layer3 = torch.tensor([float(value) for value in OutImages_1st_Layer3], dtype=torch.float32).reshape(8, 128, 52, 52)

        # Mean, Var
        Mean_1st_Layer3, Var_1st_Layer3 = Cal_mean_var.forward(OutImage_1st_Layer3)


        Beta_Layer3 = data.Beta_Dec[3]
        Gamma_Layer3 = data.Gamma_Dec[3]

        layer3_cache = BN(OutImage_1st_Layer3, Gamma_Layer3, Beta_Layer3)

        # Squeeze to remove the dimension but keeping the same data ordering
        Var_1st_Layer3 = Var_1st_Layer3.squeeze() * Gamma_Layer3
        s = time.time()
        Mean_1st_Layer3, Var_1st_Layer3 = Mean_Var_Dec2Bfloat(Mean_1st_Layer3, Var_1st_Layer3, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat : ",e-s)
        s = time.time()
        Weight_2nd_Layer3 = New_Weight_Hardware_ReOrdering_OtherLayer(128, 64, data.Weight_Bfloat[3], Mean_1st_Layer3, Var_1st_Layer3, data.Beta_Bfloat[3], Iteration="2")
        e = time.time()
        if DEBUG: print("Weight Reordering : ",e-s)    

        '''
        data_read_mean_var = "result/layer3_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer3:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")     
        '''           

        # Write DDR
        s = time.time()
        Write_DDR(data_256_32(Weight_2nd_Layer3[0]), Wr_Address=0x80006E00)
        Write_DDR(data_256_32(Weight_2nd_Layer3[1]), Wr_Address=0x90006E00)
        e = time.time()
        if DEBUG: print("Write DDR & 256bit to 32bit : ",e-s)

        layer3_end = time.time()
        layer3_process = layer3_end - layer3_start
        if DEBUG: print("Layer3 process time : ", layer3_process)
        
        resume()
        #if DEBUG: print(irq_val)

        '''
        d = Device("0000:08:00.0")
        bar = d.bar[0]

        data_read = open("result/layer3_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        d = Device("0000:08:00.0")
        bar = d.bar[2]

        data_read = open("result/layer3_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X6770000-0X66A0000)/4) ): 
            Read_Data = bar.read(0X66A0000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer3_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X6770000-0X66A0000)/4) ): 
            Read_Data = bar.read(0X166A0000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer3_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X7DFC000-0X7D2C000)/4) ): 
            Read_Data = bar.read(0X7D2C000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      
        '''

        #################################################
        #                Layer 4 Start                  #
        #################################################
        # check Layer4 IRQ
        check_irq_otherlayer()
        # self.app_instance .change_color(self.app_instance.L5_IRQ_canvas, self.app_instance.L5_IRQ, "green")
        # Layer 4
        Layer4_start = time.time()
        # Read DDR & Conver Format # 512MB
        s = time.time()
        Layer4_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x86770000, End_Address=0x867A4000)
        Layer4_1st_Iter_Image1_CH0_256 = (data_32_to_16(Layer4_1st_Iter_Image1_CH0))   
        #if DEBUG: print("ch0 image 1 : ", len(Layer4_1st_Iter_Image1_CH0))     

        Layer4_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x867A4000, End_Address=0x867D8000)
        Layer4_1st_Iter_Image2_CH0_256 = (data_32_to_16(Layer4_1st_Iter_Image2_CH0))
        #if DEBUG: print("ch0 image 2 : ", len(Layer4_1st_Iter_Image2_CH0))
        
        Layer4_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x867D8000, End_Address=0x8680C000)
        Layer4_1st_Iter_Image3_CH0_256 = (data_32_to_16(Layer4_1st_Iter_Image3_CH0))
        #if DEBUG: print("ch0 image 3 : ", len(Layer4_1st_Iter_Image3_CH0))

        Layer4_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x8680C000, End_Address=0x86840000)
        Layer4_1st_Iter_Image4_CH0_256 = (data_32_to_16(Layer4_1st_Iter_Image4_CH0))
        #if DEBUG: print("ch0 image 4 : ", len(Layer4_1st_Iter_Image4_CH0))

        Layer4_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x86840000, End_Address=0x86874000)
        Layer4_1st_Iter_Image5_CH0_256 = (data_32_to_16(Layer4_1st_Iter_Image5_CH0))
        #if DEBUG: print("ch0 image 5 : ", len(Layer4_1st_Iter_Image5_CH0))

        Layer4_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x86874000, End_Address=0x868A8000)
        Layer4_1st_Iter_Image6_CH0_256 = (data_32_to_16(Layer4_1st_Iter_Image6_CH0))
        #if DEBUG: print("ch0 image 6 : ", len(Layer4_1st_Iter_Image6_CH0))

        Layer4_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x868A8000, End_Address=0x868DC000)
        Layer4_1st_Iter_Image7_CH0_256 = (data_32_to_16(Layer4_1st_Iter_Image7_CH0))
        #if DEBUG: print("ch0 image 7 : ", len(Layer4_1st_Iter_Image7_CH0))

        Layer4_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x868DC000, End_Address=0x86910000)
        Layer4_1st_Iter_Image8_CH0_256 = (data_32_to_16(Layer4_1st_Iter_Image8_CH0))
        #if DEBUG: print("ch0 image 8 : ", len(Layer4_1st_Iter_Image8_CH0))


        Layer4_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x96770000, End_Address=0x967A4000)
        Layer4_1st_Iter_Image1_CH1_256 = (data_32_to_16(Layer4_1st_Iter_Image1_CH1))
        #if DEBUG: print("ch1 image 1 : ", len(Layer4_1st_Iter_Image1_CH1))

        Layer4_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x967A4000, End_Address=0x967D8000)
        Layer4_1st_Iter_Image2_CH1_256 = (data_32_to_16(Layer4_1st_Iter_Image2_CH1))
        #if DEBUG: print("ch1 image 2 : ", len(Layer4_1st_Iter_Image2_CH1))

        Layer4_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x967D8000, End_Address=0x9680C000)
        Layer4_1st_Iter_Image3_CH1_256 = (data_32_to_16(Layer4_1st_Iter_Image3_CH1))
        #if DEBUG: print("ch1 image 3 : ", len(Layer4_1st_Iter_Image3_CH1))

        Layer4_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x9680C000, End_Address=0x96840000)
        Layer4_1st_Iter_Image4_CH1_256 = (data_32_to_16(Layer4_1st_Iter_Image4_CH1))
        #if DEBUG: print("ch1 image 4 : ", len(Layer4_1st_Iter_Image4_CH1))

        Layer4_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x96840000, End_Address=0x96874000)
        Layer4_1st_Iter_Image5_CH1_256 = (data_32_to_16(Layer4_1st_Iter_Image5_CH1))
        #if DEBUG: print("ch1 image 5 : ", len(Layer4_1st_Iter_Image5_CH1))

        Layer4_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x96874000, End_Address=0x968A8000)
        Layer4_1st_Iter_Image6_CH1_256 = (data_32_to_16(Layer4_1st_Iter_Image6_CH1))
        #if DEBUG: print("ch1 image 6 : ", len(Layer4_1st_Iter_Image6_CH1))

        Layer4_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x968A8000, End_Address=0x968DC000)
        Layer4_1st_Iter_Image7_CH1_256 = (data_32_to_16(Layer4_1st_Iter_Image7_CH1))
        #if DEBUG: print("ch1 image 7 : ", len(Layer4_1st_Iter_Image7_CH1))

        Layer4_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x968DC000, End_Address=0x96910000)
        Layer4_1st_Iter_Image8_CH1_256 = (data_32_to_16(Layer4_1st_Iter_Image8_CH1))
        #if DEBUG: print("ch1 image 8 : ", len(Layer4_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR & 32bit to 16bit : ",e-s)


        '''
        test_out = '1st_iter_result/Layer4_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''
        
        s = time.time()
        Output_Image1_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image1_CH0_256, Layer4_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image2_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image2_CH0_256, Layer4_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image3_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image3_CH0_256, Layer4_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image4_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image4_CH0_256, Layer4_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image5_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image5_CH0_256, Layer4_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image6_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image6_CH0_256, Layer4_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image7_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image7_CH0_256, Layer4_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image8_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image8_CH0_256, Layer4_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)

        OutImages_1st_Layer4 = Output_Image1_Layer4_1st_Iter + Output_Image2_Layer4_1st_Iter + Output_Image3_Layer4_1st_Iter + Output_Image4_Layer4_1st_Iter + \
                            Output_Image5_Layer4_1st_Iter + Output_Image6_Layer4_1st_Iter + Output_Image7_Layer4_1st_Iter + Output_Image8_Layer4_1st_Iter    

        OutImage_1st_Layer4 = torch.tensor([float(value) for value in OutImages_1st_Layer4], dtype=torch.float32).reshape(8, 256, 26, 26)

        # Mean, Var
        s = time.time()
        Mean_1st_Layer4, Var_1st_Layer4 = Cal_mean_var.forward(OutImage_1st_Layer4)
        e = time.time()
        if DEBUG: print("Calculate Mean & Var : ",e-s)

        Beta_Layer4 = data.Beta_Dec[4]
        Gamma_Layer4 = data.Gamma_Dec[4]

        layer4_cache = BN(OutImage_1st_Layer4, Gamma_Layer4, Beta_Layer4)

        # Squeeze to remove the dimension but keeping the same data ordering
        Var_1st_Layer4 = Var_1st_Layer4.squeeze() * Gamma_Layer4

        s = time.time()
        Mean_1st_Layer4, Var_1st_Layer4 = Mean_Var_Dec2Bfloat(Mean_1st_Layer4, Var_1st_Layer4, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat : ",e-s)
        s = time.time()
        Weight_2nd_Layer4 = New_Weight_Hardware_ReOrdering_OtherLayer(256, 128, data.Weight_Bfloat[4], Mean_1st_Layer4, Var_1st_Layer4, data.Beta_Bfloat[4], Iteration="2")
        e = time.time()
        if DEBUG: print("Weight Reordering : ",e-s)

        '''
        data_read_mean_var = "result/layer4_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer4:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")        
        '''

        # Write DDR
        s = time.time()
        Write_DDR(data_256_32(Weight_2nd_Layer4[0]), Wr_Address=0x8001AE00)
        Write_DDR(data_256_32(Weight_2nd_Layer4[1]), Wr_Address=0x9001AE00)
        e = time.time()
        if DEBUG: print("Write DDR & 256bit to 32bit : ",e-s)

        layer4_end = time.time()
        layer4_process = layer4_end - Layer4_start
        if DEBUG: print("Layer4 process time : ", layer4_process)
        
        resume()

        '''
        d = Device("0000:08:00.0")
        bar = d.bar[0]

        data_read = open("result/layer4_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        d = Device("0000:08:00.0")
        bar = d.bar[2]

        data_read = open("result/layer4_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X6978000-0X6910000)/4) ): 
            Read_Data = bar.read(0X6910000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer4_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X6978000-0X6910000)/4) ): 
            Read_Data = bar.read(0X16910000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer4_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X7E64000-0X7DFC000)/4) ): 
            Read_Data = bar.read(0X7DFC000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      
        '''

        #################################################
        #                Layer 5 Start                  #
        #################################################
        # check Layer5 IRQ
        check_irq_otherlayer()
        # self.app_instance .change_color(self.app_instance.L6_IRQ_canvas, self.app_instance.L6_IRQ, "green")
        # Layer 5
        Layer5_start = time.time()
        s = time.time()
        # Read DDR & Conver Format # 512MB
        Layer5_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x86978000, End_Address=0x86992000)
        Layer5_1st_Iter_Image1_CH0_256 = (data_32_to_16(Layer5_1st_Iter_Image1_CH0))   
        #if DEBUG: print("ch0 image 1 : ", len(Layer5_1st_Iter_Image1_CH0))     

        Layer5_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x86992000, End_Address=0x869AC000)
        Layer5_1st_Iter_Image2_CH0_256 = (data_32_to_16(Layer5_1st_Iter_Image2_CH0))
        #if DEBUG: print("ch0 image 2 : ", len(Layer5_1st_Iter_Image2_CH0))
        
        Layer5_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x869AC000, End_Address=0x869C6000)
        Layer5_1st_Iter_Image3_CH0_256 = (data_32_to_16(Layer5_1st_Iter_Image3_CH0))
        #if DEBUG: print("ch0 image 3 : ", len(Layer5_1st_Iter_Image3_CH0))

        Layer5_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x869C6000, End_Address=0x869E0000)
        Layer5_1st_Iter_Image4_CH0_256 = (data_32_to_16(Layer5_1st_Iter_Image4_CH0))
        #if DEBUG: print("ch0 image 4 : ", len(Layer5_1st_Iter_Image4_CH0))

        Layer5_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x869E0000, End_Address=0x869FA000)
        Layer5_1st_Iter_Image5_CH0_256 = (data_32_to_16(Layer5_1st_Iter_Image5_CH0))
        #if DEBUG: print("ch0 image 5 : ", len(Layer5_1st_Iter_Image5_CH0))

        Layer5_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x869FA000, End_Address=0x86A14000)
        Layer5_1st_Iter_Image6_CH0_256 = (data_32_to_16(Layer5_1st_Iter_Image6_CH0))
        #if DEBUG: print("ch0 image 6 : ", len(Layer5_1st_Iter_Image6_CH0))

        Layer5_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x86A14000, End_Address=0x86A2E000)
        Layer5_1st_Iter_Image7_CH0_256 = (data_32_to_16(Layer5_1st_Iter_Image7_CH0))
        #if DEBUG: print("ch0 image 7 : ", len(Layer5_1st_Iter_Image7_CH0))

        Layer5_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x86A2E000, End_Address=0x86A48000)
        Layer5_1st_Iter_Image8_CH0_256 = (data_32_to_16(Layer5_1st_Iter_Image8_CH0))
        #if DEBUG: print("ch0 image 8 : ", len(Layer5_1st_Iter_Image8_CH0))


        Layer5_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x96978000, End_Address=0x96992000)
        Layer5_1st_Iter_Image1_CH1_256 = (data_32_to_16(Layer5_1st_Iter_Image1_CH1))
        #if DEBUG: print("ch1 image 1 : ", len(Layer5_1st_Iter_Image1_CH1))

        Layer5_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x96992000, End_Address=0x969AC000)
        Layer5_1st_Iter_Image2_CH1_256 = (data_32_to_16(Layer5_1st_Iter_Image2_CH1))
        #if DEBUG: print("ch1 image 2 : ", len(Layer5_1st_Iter_Image2_CH1))

        Layer5_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x969AC000, End_Address=0x969C6000)
        Layer5_1st_Iter_Image3_CH1_256 = (data_32_to_16(Layer5_1st_Iter_Image3_CH1))
        #if DEBUG: print("ch1 image 3 : ", len(Layer5_1st_Iter_Image3_CH1))

        Layer5_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x969C6000, End_Address=0x969E0000)
        Layer5_1st_Iter_Image4_CH1_256 = (data_32_to_16(Layer5_1st_Iter_Image4_CH1))
        #if DEBUG: print("ch1 image 4 : ", len(Layer5_1st_Iter_Image4_CH1))

        Layer5_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x969E0000, End_Address=0x969FA000)
        Layer5_1st_Iter_Image5_CH1_256 = (data_32_to_16(Layer5_1st_Iter_Image5_CH1))
        #if DEBUG: print("ch1 image 5 : ", len(Layer5_1st_Iter_Image5_CH1))

        Layer5_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x969FA000, End_Address=0x96A14000)
        Layer5_1st_Iter_Image6_CH1_256 = (data_32_to_16(Layer5_1st_Iter_Image6_CH1))
        #if DEBUG: print("ch1 image 6 : ", len(Layer5_1st_Iter_Image6_CH1))

        Layer5_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x96A14000, End_Address=0x96A2E000)
        Layer5_1st_Iter_Image7_CH1_256 = (data_32_to_16(Layer5_1st_Iter_Image7_CH1))
        #if DEBUG: print("ch1 image 7 : ", len(Layer5_1st_Iter_Image7_CH1))

        Layer5_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x96A2E000, End_Address=0x96A48000)
        Layer5_1st_Iter_Image8_CH1_256 = (data_32_to_16(Layer5_1st_Iter_Image8_CH1))
        #if DEBUG: print("ch1 image 8 : ", len(Layer5_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR & 32bit to 16bit : ",e-s)

        '''
        test_out = '1st_iter_result/Layer5_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''
        

        s = time.time()
        Output_Image1_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image1_CH0_256, Layer5_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image2_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image2_CH0_256, Layer5_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image3_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image3_CH0_256, Layer5_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image4_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image4_CH0_256, Layer5_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image5_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image5_CH0_256, Layer5_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image6_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image6_CH0_256, Layer5_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image7_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image7_CH0_256, Layer5_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image8_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image8_CH0_256, Layer5_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)

        OutImages_1st_Layer5 = Output_Image1_Layer5_1st_Iter + Output_Image2_Layer5_1st_Iter + Output_Image3_Layer5_1st_Iter + Output_Image4_Layer5_1st_Iter + \
                            Output_Image5_Layer5_1st_Iter + Output_Image6_Layer5_1st_Iter + Output_Image7_Layer5_1st_Iter + Output_Image8_Layer5_1st_Iter    

        OutImage_1st_Layer5 = torch.tensor([float(value) for value in OutImages_1st_Layer5], dtype=torch.float32).reshape(8, 512, 13, 13)

        # Mean, Var
        s = time.time()
        Mean_1st_Layer5, Var_1st_Layer5 = Cal_mean_var.forward(OutImage_1st_Layer5)
        e = time.time()
        if DEBUG: print("Calculate Mean & Var : ",e-s)

        Beta_Layer5 = data.Beta_Dec[5]
        Gamma_Layer5 = data.Gamma_Dec[5]

        layer5_cache = BN(OutImage_1st_Layer5, Gamma_Layer5, Beta_Layer5)

        # Squeeze to remove the dimension but keeping the same data ordering
        Var_1st_Layer5 = Var_1st_Layer5.squeeze() * Gamma_Layer5

        s =time.time()
        Mean_1st_Layer5, Var_1st_Layer5 = Mean_Var_Dec2Bfloat(Mean_1st_Layer5, Var_1st_Layer5, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat : ",e-s)
        s = time.time()
        Weight_2nd_Layer5 = New_Weight_Hardware_ReOrdering_OtherLayer(512, 256, data.Weight_Bfloat[5], Mean_1st_Layer5, Var_1st_Layer5, data.Beta_Bfloat[5], Iteration="2")
        e = time.time()
        if DEBUG: print("Weight Reordering : ",e-s)

        '''
        data_read_mean_var = "result/layer5_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer5:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")    
        '''            

        # Write DDR
        s = time.time()
        Write_DDR(data_256_32(Weight_2nd_Layer5[0]), Wr_Address=0x8006AE00)
        Write_DDR(data_256_32(Weight_2nd_Layer5[1]), Wr_Address=0x9006AE00)
        e = time.time()
        if DEBUG: print("Write DDR & 256bit to 32bit : ",e-s)

        layer5_end = time.time()
        layer5_process = layer5_end - Layer5_start
        if DEBUG: print("Layer5 process time : ", layer5_process)
        
        resume()
        #if DEBUG: print(irq_val)

        '''
        d = Device("0000:08:00.0")
        bar = d.bar[0]

        data_read = open("result/layer5_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        d = Device("0000:08:00.0")
        bar = d.bar[2]

        data_read = open("result/layer5_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X6B18000-0X6A48000)/4) ): 
            Read_Data = bar.read(0X6A48000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer5_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X6B18000-0X6A48000)/4) ): 
            Read_Data = bar.read(0X16A48000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer5_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X7F34000-0X7E64000)/4) ): 
            Read_Data = bar.read(0X7E64000 + (i*4))
            data_read.write(str(Read_Data) + "\n")     
        '''

        #################################################
        #                Layer 6 Start                  #
        #################################################
        # check Layer6 IRQ
        check_irq_otherlayer()
        # self.app_instance .change_color(self.app_instance.L7_IRQ_canvas, self.app_instance.L7_IRQ, "green")
        # Layer 6
        Layer6_start = time.time()
        s = time.time()
        # Read DDR & Conver Format # 512MB
        Layer6_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x86B18000, End_Address=0x86B4C000)
        Layer6_1st_Iter_Image1_CH0_256 = (data_32_to_16(Layer6_1st_Iter_Image1_CH0))   
        #if DEBUG: print("ch0 image 1 : ", len(Layer6_1st_Iter_Image1_CH0))     

        Layer6_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x86B4C000, End_Address=0x86B80000)
        Layer6_1st_Iter_Image2_CH0_256 = (data_32_to_16(Layer6_1st_Iter_Image2_CH0))
        #if DEBUG: print("ch0 image 2 : ", len(Layer6_1st_Iter_Image2_CH0))
        
        Layer6_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x86B80000, End_Address=0x86BB4000)
        Layer6_1st_Iter_Image3_CH0_256 = (data_32_to_16(Layer6_1st_Iter_Image3_CH0))
        #if DEBUG: print("ch0 image 3 : ", len(Layer6_1st_Iter_Image3_CH0))

        Layer6_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x86BB4000, End_Address=0x86BE8000)
        Layer6_1st_Iter_Image4_CH0_256 = (data_32_to_16(Layer6_1st_Iter_Image4_CH0))
        #if DEBUG: print("ch0 image 4 : ", len(Layer6_1st_Iter_Image4_CH0))

        Layer6_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x86BE8000, End_Address=0x86C1C000)
        Layer6_1st_Iter_Image5_CH0_256 = (data_32_to_16(Layer6_1st_Iter_Image5_CH0))
        #if DEBUG: print("ch0 image 5 : ", len(Layer6_1st_Iter_Image5_CH0))

        Layer6_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x86C1C000, End_Address=0x86C50000)
        Layer6_1st_Iter_Image6_CH0_256 = (data_32_to_16(Layer6_1st_Iter_Image6_CH0))
        #if DEBUG: print("ch0 image 6 : ", len(Layer6_1st_Iter_Image6_CH0))

        Layer6_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x86C50000, End_Address=0x86C84000)
        Layer6_1st_Iter_Image7_CH0_256 = (data_32_to_16(Layer6_1st_Iter_Image7_CH0))
        #if DEBUG: print("ch0 image 7 : ", len(Layer6_1st_Iter_Image7_CH0))

        Layer6_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x86C84000, End_Address=0x86CB8000)
        Layer6_1st_Iter_Image8_CH0_256 = (data_32_to_16(Layer6_1st_Iter_Image8_CH0))
        #if DEBUG: print("ch0 image 8 : ", len(Layer6_1st_Iter_Image8_CH0))


        Layer6_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x96B18000, End_Address=0x96B4C000)
        Layer6_1st_Iter_Image1_CH1_256 = (data_32_to_16(Layer6_1st_Iter_Image1_CH1))
        #if DEBUG: print("ch1 image 1 : ", len(Layer6_1st_Iter_Image1_CH1))

        Layer6_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x96B4C000, End_Address=0x96B80000)
        Layer6_1st_Iter_Image2_CH1_256 = (data_32_to_16(Layer6_1st_Iter_Image2_CH1))
        #if DEBUG: print("ch1 image 2 : ", len(Layer6_1st_Iter_Image2_CH1))

        Layer6_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x96B80000, End_Address=0x96BB4000)
        Layer6_1st_Iter_Image3_CH1_256 = (data_32_to_16(Layer6_1st_Iter_Image3_CH1))
        #if DEBUG: print("ch1 image 3 : ", len(Layer6_1st_Iter_Image3_CH1))

        Layer6_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x96BB4000, End_Address=0x96BE8000)
        Layer6_1st_Iter_Image4_CH1_256 = (data_32_to_16(Layer6_1st_Iter_Image4_CH1))
        #if DEBUG: print("ch1 image 4 : ", len(Layer6_1st_Iter_Image4_CH1))

        Layer6_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x96BE8000, End_Address=0x96C1C000)
        Layer6_1st_Iter_Image5_CH1_256 = (data_32_to_16(Layer6_1st_Iter_Image5_CH1))
        #if DEBUG: print("ch1 image 5 : ", len(Layer6_1st_Iter_Image5_CH1))

        Layer6_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x96C1C000, End_Address=0x96C50000)
        Layer6_1st_Iter_Image6_CH1_256 = (data_32_to_16(Layer6_1st_Iter_Image6_CH1))
        #if DEBUG: print("ch1 image 6 : ", len(Layer6_1st_Iter_Image6_CH1))

        Layer6_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x96C50000, End_Address=0x96C84000)
        Layer6_1st_Iter_Image7_CH1_256 = (data_32_to_16(Layer6_1st_Iter_Image7_CH1))
        #if DEBUG: print("ch1 image 7 : ", len(Layer6_1st_Iter_Image7_CH1))

        Layer6_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x96C84000, End_Address=0x96CB8000)
        Layer6_1st_Iter_Image8_CH1_256 = (data_32_to_16(Layer6_1st_Iter_Image8_CH1))
        #if DEBUG: print("ch1 image 8 : ", len(Layer6_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR & 32bit to 16bit : ",e-s)


        '''
        test_out = '1st_iter_result/Layer6_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''
        
        s = time.time()
        Output_Image1_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image1_CH0_256, Layer6_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image2_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image2_CH0_256, Layer6_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image3_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image3_CH0_256, Layer6_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image4_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image4_CH0_256, Layer6_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image5_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image5_CH0_256, Layer6_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image6_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image6_CH0_256, Layer6_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image7_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image7_CH0_256, Layer6_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image8_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image8_CH0_256, Layer6_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)

        OutImages_1st_Layer6 = Output_Image1_Layer6_1st_Iter + Output_Image2_Layer6_1st_Iter + Output_Image3_Layer6_1st_Iter + Output_Image4_Layer6_1st_Iter + \
                            Output_Image5_Layer6_1st_Iter + Output_Image6_Layer6_1st_Iter + Output_Image7_Layer6_1st_Iter + Output_Image8_Layer6_1st_Iter    

        OutImage_1st_Layer6 = torch.tensor([float(value) for value in OutImages_1st_Layer6], dtype=torch.float32).reshape(8, 1024, 13, 13)

        # Mean, Var
        Mean_1st_Layer6, Var_1st_Layer6 = Cal_mean_var.forward(OutImage_1st_Layer6)
        
        Beta_Layer6 = data.Beta_Dec[6]
        Gamma_Layer6 = data.Gamma_Dec[6]

        layer6_cache = BN(OutImage_1st_Layer6, Gamma_Layer6, Beta_Layer6)

        # Squeeze to remove the dimension but keeping the same data ordering
        Var_1st_Layer6 = Var_1st_Layer6.squeeze() * Gamma_Layer6

        s = time.time()
        Mean_1st_Layer6, Var_1st_Layer6 = Mean_Var_Dec2Bfloat(Mean_1st_Layer6, Var_1st_Layer6, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat : ",e-s)
        s = time.time()
        Weight_2nd_Layer6 = New_Weight_Hardware_ReOrdering_OtherLayer(1024, 512, data.Weight_Bfloat[6], Mean_1st_Layer6, Var_1st_Layer6, data.Beta_Bfloat[6], Iteration="2")
        e = time.time()
        if DEBUG: print("Weight Reordering : ",e-s)

        '''
        data_read_mean_var = "result/layer6_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer6:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")        
        '''        

        # Write DDR
        s = time.time()
        Write_DDR(data_256_32(Weight_2nd_Layer6[0]), Wr_Address=0x801AAE00)
        Write_DDR(data_256_32(Weight_2nd_Layer6[1]), Wr_Address=0x901AAE00)
        e = time.time()
        if DEBUG: print("Write DDR & 256bit to 32bit : ",e-s)

        layer6_end = time.time()
        layer6_process = layer6_end - Layer6_start
        if DEBUG: print("Layer6 process time : ", layer6_process)

        resume()

        '''
        d = Device("0000:08:00.0")
        bar = d.bar[0]

        data_read = open("result/layer6_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        d = Device("0000:08:00.0")
        bar = d.bar[2]

        data_read = open("result/layer6_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X6E58000-0X6CB8000)/4) ): 
            Read_Data = bar.read(0X6CB8000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer6_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X6E58000-0X6CB8000)/4) ): 
            Read_Data = bar.read(0X16CB8000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer6_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X80D4000-0X7F34000)/4) ): 
            Read_Data = bar.read(0X7F34000 + (i*4))
            data_read.write(str(Read_Data) + "\n")       
        '''

        #################################################
        #                Layer 7 Start                  #
        #################################################
        # check Layer7 IRQ
        check_irq_otherlayer()
        # self.app_instance .change_color(self.app_instance.L8_IRQ_canvas, self.app_instance.L8_IRQ, "green")
        # Layer 7
        Layer7_start = time.time()
        s = time.time()
        # Read DDR & Conver Format # 512MB
        # if DEBUG: print("Read DDR")
        Layer7_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x86E58000, End_Address=0x86E8C000)
        Layer7_1st_Iter_Image1_CH0_256 = (data_32_to_16(Layer7_1st_Iter_Image1_CH0))   
        #if DEBUG: print("ch0 image 1 : ", len(Layer7_1st_Iter_Image1_CH0))     

        Layer7_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x86E8C000, End_Address=0x86EC0000)
        Layer7_1st_Iter_Image2_CH0_256 = (data_32_to_16(Layer7_1st_Iter_Image2_CH0))
        #if DEBUG: print("ch0 image 2 : ", len(Layer7_1st_Iter_Image2_CH0))
        
        Layer7_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x86EC0000, End_Address=0x86EF4000)
        Layer7_1st_Iter_Image3_CH0_256 = (data_32_to_16(Layer7_1st_Iter_Image3_CH0))
        #if DEBUG: print("ch0 image 3 : ", len(Layer7_1st_Iter_Image3_CH0))

        Layer7_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x86EF4000, End_Address=0x86F28000)
        Layer7_1st_Iter_Image4_CH0_256 = (data_32_to_16(Layer7_1st_Iter_Image4_CH0))
        #if DEBUG: print("ch0 image 4 : ", len(Layer7_1st_Iter_Image4_CH0))

        Layer7_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x86F28000, End_Address=0x86F5C000)
        Layer7_1st_Iter_Image5_CH0_256 = (data_32_to_16(Layer7_1st_Iter_Image5_CH0))
        #if DEBUG: print("ch0 image 5 : ", len(Layer7_1st_Iter_Image5_CH0))

        Layer7_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x86F5C000, End_Address=0x86F90000)
        Layer7_1st_Iter_Image6_CH0_256 = (data_32_to_16(Layer7_1st_Iter_Image6_CH0))
        #if DEBUG: print("ch0 image 6 : ", len(Layer7_1st_Iter_Image6_CH0))

        Layer7_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x86F90000, End_Address=0x86FC4000)
        Layer7_1st_Iter_Image7_CH0_256 = (data_32_to_16(Layer7_1st_Iter_Image7_CH0))
        #if DEBUG: print("ch0 image 7 : ", len(Layer7_1st_Iter_Image7_CH0))

        Layer7_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x86FC4000, End_Address=0x86FF8000)
        Layer7_1st_Iter_Image8_CH0_256 = (data_32_to_16(Layer7_1st_Iter_Image8_CH0))
        #if DEBUG: print("ch0 image 8 : ", len(Layer7_1st_Iter_Image8_CH0))


        Layer7_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x96E58000, End_Address=0x96E8C000)
        Layer7_1st_Iter_Image1_CH1_256 = (data_32_to_16(Layer7_1st_Iter_Image1_CH1))
        #if DEBUG: print("ch1 image 1 : ", len(Layer7_1st_Iter_Image1_CH1))

        Layer7_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x96E8C000, End_Address=0x96EC0000)
        Layer7_1st_Iter_Image2_CH1_256 = (data_32_to_16(Layer7_1st_Iter_Image2_CH1))
        #if DEBUG: print("ch1 image 2 : ", len(Layer7_1st_Iter_Image2_CH1))

        Layer7_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x96EC0000, End_Address=0x96EF4000)
        Layer7_1st_Iter_Image3_CH1_256 = (data_32_to_16(Layer7_1st_Iter_Image3_CH1))
        #if DEBUG: print("ch1 image 3 : ", len(Layer7_1st_Iter_Image3_CH1))

        Layer7_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x96EF4000, End_Address=0x96F28000)
        Layer7_1st_Iter_Image4_CH1_256 = (data_32_to_16(Layer7_1st_Iter_Image4_CH1))
        #if DEBUG: print("ch1 image 4 : ", len(Layer7_1st_Iter_Image4_CH1))

        Layer7_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x96F28000, End_Address=0x96F5C000)
        Layer7_1st_Iter_Image5_CH1_256 = (data_32_to_16(Layer7_1st_Iter_Image5_CH1))
        #if DEBUG: print("ch1 image 5 : ", len(Layer7_1st_Iter_Image5_CH1))

        Layer7_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x96F5C000, End_Address=0x96F90000)
        Layer7_1st_Iter_Image6_CH1_256 = (data_32_to_16(Layer7_1st_Iter_Image6_CH1))
        #if DEBUG: print("ch1 image 6 : ", len(Layer7_1st_Iter_Image6_CH1))

        Layer7_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x96F90000, End_Address=0x96FC4000)
        Layer7_1st_Iter_Image7_CH1_256 = (data_32_to_16(Layer7_1st_Iter_Image7_CH1))
        #if DEBUG: print("ch1 image 7 : ", len(Layer7_1st_Iter_Image7_CH1))

        Layer7_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x96FC4000, End_Address=0x96FF8000)
        Layer7_1st_Iter_Image8_CH1_256 = (data_32_to_16(Layer7_1st_Iter_Image8_CH1))
        #if DEBUG: print("ch1 image 8 : ", len(Layer7_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR Time : ",e-s)


        '''
        test_out = '1st_iter_result/Layer7_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''
        s = time.time()
        Output_Image1_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image1_CH0_256, Layer7_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image2_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image2_CH0_256, Layer7_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image3_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image3_CH0_256, Layer7_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image4_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image4_CH0_256, Layer7_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image5_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image5_CH0_256, Layer7_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image6_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image6_CH0_256, Layer7_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image7_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image7_CH0_256, Layer7_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image8_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image8_CH0_256, Layer7_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)

        OutImages_1st_Layer7 = Output_Image1_Layer7_1st_Iter + Output_Image2_Layer7_1st_Iter + Output_Image3_Layer7_1st_Iter + Output_Image4_Layer7_1st_Iter + \
                            Output_Image5_Layer7_1st_Iter + Output_Image6_Layer7_1st_Iter + Output_Image7_Layer7_1st_Iter + Output_Image8_Layer7_1st_Iter    

        OutImage_1st_Layer7 = torch.tensor([float(value) for value in OutImages_1st_Layer7], dtype=torch.float32).reshape(8, 1024, 13, 13)
        e = time.time()
        # if DEBUG: print("OutFmap_Bfloat2Dec Convert Time : ", e-s)

        # Mean, Var
        s = time.time()
        Mean_1st_Layer7, Var_1st_Layer7 = Cal_mean_var.forward(OutImage_1st_Layer7)
        e = time.time()
        if DEBUG: print("Calculate Mean & Var Time : ",e-s)

        Beta_Layer7 = data.Beta_Dec[7]
        Gamma_Layer7 = data.Gamma_Dec[7]

        layer7_cache = BN(OutImage_1st_Layer7, Gamma_Layer7, Beta_Layer7)

        # Squeeze to remove the dimension but keeping the same data ordering
        Var_1st_Layer7 = Var_1st_Layer7.squeeze() * Gamma_Layer7

        s = time.time()
        Mean_1st_Layer7, Var_1st_Layer7 = Mean_Var_Dec2Bfloat(Mean_1st_Layer7, Var_1st_Layer7, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat : ",e-s)
        s = time.time()
        Weight_2nd_Layer7 = New_Weight_Hardware_ReOrdering_OtherLayer(1024, 1024, data.Weight_Bfloat[7], Mean_1st_Layer7, Var_1st_Layer7, data.Beta_Bfloat[7], Iteration="2")
        e = time.time()
        if DEBUG: print("New_Weight_Hardware_ReOrdering_OtherLayer Time : ", e-s)

        '''
        data_read_mean_var = "result/layer7_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer7:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")        
        
        '''

        # Write DDR
        s = time.time()
        Write_DDR(data_256_32(Weight_2nd_Layer7[0]), Wr_Address=0x806AAE00)
        Write_DDR(data_256_32(Weight_2nd_Layer7[1]), Wr_Address=0x906AAE00)
        e = time.time()
        if DEBUG: print("256 to 32 & Write DDR Time : ",e-s)

        layer7_end = time.time()
        layer7_process = layer7_end - Layer7_start
        if DEBUG: print("Layer7 process time : ", layer7_process)
        
        resume()

        '''
        d = Device("0000:08:00.0")
        bar = d.bar[0]

        data_read = open("result/layer7_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        d = Device("0000:08:00.0")
        bar = d.bar[2]

        data_read = open("result/layer7_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X7198000-0X6FF8000)/4) ): 
            Read_Data = bar.read(0X6FF8000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer7_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X7198000-0X6FF8000)/4) ): 
            Read_Data = bar.read(0X16FF8000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer7_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X8274000-0X80D4000)/4) ): 
            Read_Data = bar.read(0X80D4000 + (i*4))
            data_read.write(str(Read_Data) + "\n")     
        '''    
        
        end = time.time()
        process_time = (end-start)/60
        # if DEBUG: print(f'Whole Process: {process_time} mn')
        #################################################
        #                Layer 8 Start                  #
        #################################################
        check_irq_otherlayer()
        # self.app_instance .change_color(self.app_instance.L9_IRQ_canvas, self.app_instance.L9_IRQ, "green")
        layer8_start = time.time()
        s = time.time()
        # Post-Processing Pre-Defined Conditions
        #Post_Start_Signal = "1"

        # OutputImage from Hardware

        # Post Processing
        #if Post_Start_Signal == "1" or Post_Start_Signal == "1".zfill(4) or Post_Start_Signal == "1".zfill(16):  

        # Layer 8
        
        # Read DDR & Conver Format # 512MB
        
        Layer8_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x87198000, End_Address=0x8719E800)
        Layer8_1st_Iter_Image1_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image1_CH0))   
        #if DEBUG: print("ch0 image 1 : ", len(Layer8_1st_Iter_Image1_CH0))     

        Layer8_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x8719E800, End_Address=0x871A5000)
        Layer8_1st_Iter_Image2_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image2_CH0))
        #if DEBUG: print("ch0 image 2 : ", len(Layer8_1st_Iter_Image2_CH0))

        Layer8_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x871A5000, End_Address=0x871AB800)
        Layer8_1st_Iter_Image3_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image3_CH0))
        #if DEBUG: print("ch0 image 3 : ", len(Layer8_1st_Iter_Image3_CH0))

        Layer8_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x871AB800, End_Address=0x871B2000)
        Layer8_1st_Iter_Image4_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image4_CH0))
        #if DEBUG: print("ch0 image 4 : ", len(Layer8_1st_Iter_Image4_CH0))

        Layer8_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x871B2000, End_Address=0x871B8800)
        Layer8_1st_Iter_Image5_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image5_CH0))
        #if DEBUG: print("ch0 image 5 : ", len(Layer8_1st_Iter_Image5_CH0))

        Layer8_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x871B8800, End_Address=0x871BF000)
        Layer8_1st_Iter_Image6_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image6_CH0))
        #if DEBUG: print("ch0 image 6 : ", len(Layer8_1st_Iter_Image6_CH0))

        Layer8_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x871BF000, End_Address=0x871C5800)
        Layer8_1st_Iter_Image7_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image7_CH0))
        #if DEBUG: print("ch0 image 7 : ", len(Layer8_1st_Iter_Image7_CH0))

        Layer8_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x871C5800, End_Address=0x871CC000)
        Layer8_1st_Iter_Image8_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image8_CH0))
        #if DEBUG: print("ch0 image 8 : ", len(Layer8_1st_Iter_Image8_CH0))


        Layer8_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x97198000, End_Address=0x9719E800)
        Layer8_1st_Iter_Image1_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image1_CH1))   
        #if DEBUG: print("ch1 image 1 : ", len(Layer8_1st_Iter_Image1_CH1))     

        Layer8_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x9719E800, End_Address=0x971A5000)
        Layer8_1st_Iter_Image2_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image2_CH1))
        #if DEBUG: print("ch1 image 2 : ", len(Layer8_1st_Iter_Image2_CH1))

        Layer8_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x971A5000, End_Address=0x971AB800)
        Layer8_1st_Iter_Image3_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image3_CH1))
        #if DEBUG: print("ch1 image 3 : ", len(Layer8_1st_Iter_Image3_CH1))

        Layer8_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x971AB800, End_Address=0x971B2000)
        Layer8_1st_Iter_Image4_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image4_CH1))
        #if DEBUG: print("ch1 image 4 : ", len(Layer8_1st_Iter_Image4_CH1))

        Layer8_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x971B2000, End_Address=0x971B8800)
        Layer8_1st_Iter_Image5_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image5_CH1))
        #if DEBUG: print("ch1 image 5 : ", len(Layer8_1st_Iter_Image5_CH1))

        Layer8_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x971B8800, End_Address=0x971BF000)
        Layer8_1st_Iter_Image6_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image6_CH1))
        #if DEBUG: print("ch1 image 6 : ", len(Layer8_1st_Iter_Image6_CH1))

        Layer8_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x971BF000, End_Address=0x971C5800)
        Layer8_1st_Iter_Image7_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image7_CH1))
        #if DEBUG: print("ch1 image 7 : ", len(Layer8_1st_Iter_Image7_CH1))

        Layer8_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x971C5800, End_Address=0x971CC000)
        Layer8_1st_Iter_Image8_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image8_CH1))
        #if DEBUG: print("ch1 image 8 : ", len(Layer8_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR & 32bit to 16bit : ",e-s)

        Output_Image1 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image1_CH0_256, Layer8_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image2 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image2_CH0_256, Layer8_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image3 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image3_CH0_256, Layer8_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image4 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image4_CH0_256, Layer8_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image5 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image5_CH0_256, Layer8_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image6 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image6_CH0_256, Layer8_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image7 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image7_CH0_256, Layer8_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image8 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image8_CH0_256, Layer8_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Layer8 = Output_Image1 + Output_Image2 + Output_Image3 + Output_Image4 + \
                        Output_Image5 + Output_Image6 + Output_Image7 + Output_Image8

        Float_OutputImage = [np.float32(x) for x in Output_Layer8]
        Float_OutputImage = Float_OutputImage[0:(8*125*(13**2))]
        Output_Layer8 = torch.tensor(Float_OutputImage, requires_grad=True).reshape(8,125, 13, 13)
        
        if DEBUG2:
            Layer8_check_Image1_CH0 = Read_DDR(Rd_Address=0x86FF8000, End_Address=0x8702C000)
            Layer8_check_Image1_CH0_256 = (data_32_to_16(Layer8_check_Image1_CH0))   
            #if DEBUG: print("ch0 image 1 : ", len(Layer8_check_Image1_CH0))     

            Layer8_check_Image2_CH0 = Read_DDR(Rd_Address=0x8702C000, End_Address=0x87060000)
            Layer8_check_Image2_CH0_256 = (data_32_to_16(Layer8_check_Image2_CH0))
            #if DEBUG: print("ch0 image 2 : ", len(Layer8_check_Image2_CH0))

            Layer8_check_Image3_CH0 = Read_DDR(Rd_Address=0x87060000, End_Address=0x87094000)
            Layer8_check_Image3_CH0_256 = (data_32_to_16(Layer8_check_Image3_CH0))
            #if DEBUG: print("ch0 image 3 : ", len(Layer8_check_Image3_CH0))

            Layer8_check_Image4_CH0 = Read_DDR(Rd_Address=0x87094000, End_Address=0x870C8000)
            Layer8_check_Image4_CH0_256 = (data_32_to_16(Layer8_check_Image4_CH0))
            #if DEBUG: print("ch0 image 4 : ", len(Layer8_check_Image4_CH0))

            Layer8_check_Image5_CH0 = Read_DDR(Rd_Address=0x870C8000, End_Address=0x870FC000)
            Layer8_check_Image5_CH0_256 = (data_32_to_16(Layer8_check_Image5_CH0))
            #if DEBUG: print("ch0 image 5 : ", len(Layer8_check_Image5_CH0))

            Layer8_check_Image6_CH0 = Read_DDR(Rd_Address=0x870FC000, End_Address=0x87130000)
            Layer8_check_Image6_CH0_256 = (data_32_to_16(Layer8_check_Image6_CH0))
            #if DEBUG: print("ch0 image 6 : ", len(Layer8_check_Image6_CH0))

            Layer8_check_Image7_CH0 = Read_DDR(Rd_Address=0x87130000, End_Address=0x87164000)
            Layer8_check_Image7_CH0_256 = (data_32_to_16(Layer8_check_Image7_CH0))
            #if DEBUG: print("ch0 image 7 : ", len(Layer8_check_Image7_CH0))

            Layer8_check_Image8_CH0 = Read_DDR(Rd_Address=0x87164000, End_Address=0x87198000)
            Layer8_check_Image8_CH0_256 = (data_32_to_16(Layer8_check_Image8_CH0))
            #if DEBUG: print("ch0 image 8 : ", len(Layer8_check_Image8_CH0))


            Layer8_check_Image1_CH1 = Read_DDR(Rd_Address=0x96FF8000, End_Address=0x9702C000)
            Layer8_check_Image1_CH1_256 = (data_32_to_16(Layer8_check_Image1_CH1))   
            #if DEBUG: print("ch1 image 1 : ", len(Layer8_check_Image1_CH1))     

            Layer8_check_Image2_CH1 = Read_DDR(Rd_Address=0x9702C000, End_Address=0x97060000)
            Layer8_check_Image2_CH1_256 = (data_32_to_16(Layer8_check_Image2_CH1))
            #if DEBUG: print("ch1 image 2 : ", len(Layer8_check_Image2_CH1))

            Layer8_check_Image3_CH1 = Read_DDR(Rd_Address=0x97060000, End_Address=0x97094000)
            Layer8_check_Image3_CH1_256 = (data_32_to_16(Layer8_check_Image3_CH1))
            #if DEBUG: print("ch1 image 3 : ", len(Layer8_check_Image3_CH1))

            Layer8_check_Image4_CH1 = Read_DDR(Rd_Address=0x97094000, End_Address=0x970C8000)
            Layer8_check_Image4_CH1_256 = (data_32_to_16(Layer8_check_Image4_CH1))
            #if DEBUG: print("ch1 image 4 : ", len(Layer8_check_Image4_CH1))

            Layer8_check_Image5_CH1 = Read_DDR(Rd_Address=0x970C8000, End_Address=0x970FC000)
            Layer8_check_Image5_CH1_256 = (data_32_to_16(Layer8_check_Image5_CH1))
            #if DEBUG: print("ch1 image 5 : ", len(Layer8_check_Image5_CH1))

            Layer8_check_Image6_CH1 = Read_DDR(Rd_Address=0x970FC000, End_Address=0x97130000)
            Layer8_check_Image6_CH1_256 = (data_32_to_16(Layer8_check_Image6_CH1))
            #if DEBUG: print("ch1 image 6 : ", len(Layer8_check_Image6_CH1))

            Layer8_check_Image7_CH1 = Read_DDR(Rd_Address=0x97130000, End_Address=0x97164000)
            Layer8_check_Image7_CH1_256 = (data_32_to_16(Layer8_check_Image7_CH1))
            #if DEBUG: print("ch1 image 7 : ", len(Layer8_check_Image7_CH1))

            Layer8_check_Image8_CH1 = Read_DDR(Rd_Address=0x97164000, End_Address=0x97198000)
            Layer8_check_Image8_CH1_256 = (data_32_to_16(Layer8_check_Image8_CH1))
            #if DEBUG: print("ch1 image 8 : ", len(Layer8_check_Image8_CH1))            
            # Read_OutFmap_Bfloat2Dec
            # OutFmap_Layer8_BFPtoDec
            Output8_Image1 = Read_OutFmap_Bfloat2Dec(Layer8_check_Image1_CH0_256, Layer8_check_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
            Output8_Image2 = Read_OutFmap_Bfloat2Dec(Layer8_check_Image2_CH0_256, Layer8_check_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
            Output8_Image3 = Read_OutFmap_Bfloat2Dec(Layer8_check_Image3_CH0_256, Layer8_check_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
            Output8_Image4 = Read_OutFmap_Bfloat2Dec(Layer8_check_Image4_CH0_256, Layer8_check_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
            Output8_Image5 = Read_OutFmap_Bfloat2Dec(Layer8_check_Image5_CH0_256, Layer8_check_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
            Output8_Image6 = Read_OutFmap_Bfloat2Dec(Layer8_check_Image6_CH0_256, Layer8_check_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
            Output8_Image7 = Read_OutFmap_Bfloat2Dec(Layer8_check_Image7_CH0_256, Layer8_check_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
            Output8_Image8 = Read_OutFmap_Bfloat2Dec(Layer8_check_Image8_CH0_256, Layer8_check_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
            Output8_Layer8 = Output8_Image1 + Output8_Image2 + Output8_Image3 + Output8_Image4 + \
                            Output8_Image5 + Output8_Image6 + Output8_Image7 + Output8_Image8

            Float_Output8Image = [np.float32(x) for x in Output8_Layer8]
            Float_Output8Image = Float_Output8Image[0:(8*125*(13**2))]
            input_Layer8 = torch.tensor(Float_Output8Image, requires_grad=True).reshape(8,125, 13, 13)
            Save_File(input_Layer8, "result/Layer_8_input")   
            
        
        if DEBUG2 : Save_File(Output_Layer8, "result/Layer_8_Forward")   
        if DEBUG2 : Save_File(data.Bias_Dec[8],   "result/Layer_8_Forward_Bias_Before_Weight_Update")
        if DEBUG2 : Save_File(data.Weight_Dec[8], "result/Layer_8_Forward_weight_Before_Weight_Update")
        
        return Output_Layer8


    def Post_Processing(self, data, gt_boxes, gt_classes, num_boxes):
        # check Layer8 IRQ

        '''
        test_out = '1st_iter_result/Layer8_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer8_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer8_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer8_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer8_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer8_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer8_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer8_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer8_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer8_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer8_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer8_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer8_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer8_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer8_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer8_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer8_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer8_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer8_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer8_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer8_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer8_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer8_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer8_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer8_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer8_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer8_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer8_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer8_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer8_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer8_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer8_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''
        

        # if data.Mode == "Training":
        PostProcessing = Post_Processing(Mode="Training",
                    Brain_Floating_Point=data.Brain_Floating_Point,
                    Exponent_Bits=Exponent_Bits,
                    Mantissa_Bits=Mantissa_Bits,
                    Output_Layer8=Output_Layer8,
                    # OutImage1_Data_CH1=Layer8_1st_Iter_Image1_CH1_256,
                    # OutImage2_Data_CH0=Layer8_1st_Iter_Image2_CH0_256,
                    # OutImage2_Data_CH1=Layer8_1st_Iter_Image2_CH1_256,
                    # OutImage3_Data_CH0=Layer8_1st_Iter_Image3_CH0_256,
                    # OutImage3_Data_CH1=Layer8_1st_Iter_Image3_CH1_256,
                    # OutImage4_Data_CH0=Layer8_1st_Iter_Image4_CH0_256,
                    # OutImage4_Data_CH1=Layer8_1st_Iter_Image4_CH1_256,
                    # OutImage5_Data_CH0=Layer8_1st_Iter_Image5_CH0_256,
                    # OutImage5_Data_CH1=Layer8_1st_Iter_Image5_CH1_256,
                    # OutImage6_Data_CH0=Layer8_1st_Iter_Image6_CH0_256,
                    # OutImage6_Data_CH1=Layer8_1st_Iter_Image6_CH1_256,
                    # OutImage7_Data_CH0=Layer8_1st_Iter_Image7_CH0_256,
                    # OutImage7_Data_CH1=Layer8_1st_Iter_Image7_CH1_256,
                    # OutImage8_Data_CH0=Layer8_1st_Iter_Image8_CH0_256,
                    # OutImage8_Data_CH1=Layer8_1st_Iter_Image8_CH1_256
                    )
        s = time.time()
        Loss, Loss_Gradient = PostProcessing.PostProcessing(gt_boxes, gt_classes, num_boxes)
        e = time.time()
        if DEBUG: print("Calculate Loss : ",e-s)
        # if DEBUG: print(Loss)
        #if DEBUG: print(Loss_Gradient)
        
        output_file1 = "loss.txt"
        with open(output_file1, mode="a") as output_file_1:
            output_file_1.write(str(Loss) + "\n")
        output_file2 = "loss_gradient.txt"
        with open(output_file2, mode="w") as output_file_2:
            for item in (Loss_Gradient):
                output_file_2.write(str(item) + "\n")        
        output_file_1.close()
        output_file_2.close()
        
        return Loss, Loss_Gradient
    
   
            
    def Forward_infer_test(self,data):
        global layer0_cache, layer1_cache, layer2_cache, layer3_cache, layer4_cache, layer5_cache, layer6_cache, layer7_cache     
        start = time.time()
        #################################################
        #                Layer 0 Start                  #
        #################################################       
        # layer0 capture interrupt
        # TODO: Moved check_irq func to main
        # check_irq_layer0() 
        # self.app_instance.change_color(self.app_instance.L1_IRQ_canvas, self.app_instance.L1_IRQ, "green")
        global OutImage_1st_Layer0, OutImage_1st_Layer1, OutImage_1st_Layer2, OutImage_1st_Layer3, OutImage_1st_Layer4,\
        OutImage_1st_Layer5, OutImage_1st_Layer5, OutImage_1st_Layer7, OutImage_1st_Layer8, Output_Layer8, Bias_Grad
        # Layer 0
        # Read DDR & Conver Format # 512MB
        layer0_start = time.time()

        s = time.time()
        Layer0_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x83E00000, End_Address=0x83ED0000)
        Layer0_1st_Iter_Image1_CH0_256 = data_32_to_16(Layer0_1st_Iter_Image1_CH0) 
        #if DEBUG: print("ch0 image 1 : ", len(Layer0_1st_Iter_Image1_CH0)) 

        Layer0_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x83ED0000, End_Address=0x83FA0000)
        Layer0_1st_Iter_Image2_CH0_256 = data_32_to_16(Layer0_1st_Iter_Image2_CH0)
        #if DEBUG: print("ch0 image 2 : ", len(Layer0_1st_Iter_Image2_CH0))
        
        Layer0_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x83FA0000, End_Address=0x84070000)
        Layer0_1st_Iter_Image3_CH0_256 = data_32_to_16(Layer0_1st_Iter_Image3_CH0)
        #if DEBUG: print("ch0 image 3 : ", len(Layer0_1st_Iter_Image3_CH0))

        Layer0_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x84070000, End_Address=0x84140000)
        Layer0_1st_Iter_Image4_CH0_256 = data_32_to_16(Layer0_1st_Iter_Image4_CH0)
        #if DEBUG: print("ch0 image 4 : ", len(Layer0_1st_Iter_Image4_CH0))

        Layer0_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x84140000, End_Address=0x84210000)
        Layer0_1st_Iter_Image5_CH0_256 = data_32_to_16(Layer0_1st_Iter_Image5_CH0)
        #if DEBUG: print("ch0 image 5 : ", len(Layer0_1st_Iter_Image5_CH0))

        Layer0_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x84210000, End_Address=0x842E0000)
        Layer0_1st_Iter_Image6_CH0_256 = data_32_to_16(Layer0_1st_Iter_Image6_CH0)
        #if DEBUG: print("ch0 image 6 : ", len(Layer0_1st_Iter_Image6_CH0))

        Layer0_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x842E0000, End_Address=0x843B0000)
        Layer0_1st_Iter_Image7_CH0_256 = data_32_to_16(Layer0_1st_Iter_Image7_CH0)
        #if DEBUG: print("ch0 image 7 : ", len(Layer0_1st_Iter_Image7_CH0))

        Layer0_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x843B0000, End_Address=0x84480000)
        Layer0_1st_Iter_Image8_CH0_256 = data_32_to_16(Layer0_1st_Iter_Image8_CH0)
        #if DEBUG: print("ch0 image 8 : ", len(Layer0_1st_Iter_Image8_CH0))


        Layer0_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x93E00000, End_Address=0x93ED0000)
        Layer0_1st_Iter_Image1_CH1_256 = data_32_to_16(Layer0_1st_Iter_Image1_CH1)
        #if DEBUG: print("ch1 image 1 : ", len(Layer0_1st_Iter_Image1_CH1))

        Layer0_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x93ED0000, End_Address=0x93FA0000)
        Layer0_1st_Iter_Image2_CH1_256 = data_32_to_16(Layer0_1st_Iter_Image2_CH1)
        #if DEBUG: print("ch1 image 2 : ", len(Layer0_1st_Iter_Image2_CH1))

        Layer0_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x93FA0000, End_Address=0x94070000)
        Layer0_1st_Iter_Image3_CH1_256 = data_32_to_16(Layer0_1st_Iter_Image3_CH1)
        #if DEBUG: print("ch1 image 3 : ", len(Layer0_1st_Iter_Image3_CH1))

        Layer0_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x94070000, End_Address=0x94140000)
        Layer0_1st_Iter_Image4_CH1_256 = data_32_to_16(Layer0_1st_Iter_Image4_CH1)
        #if DEBUG: print("ch1 image 4 : ", len(Layer0_1st_Iter_Image4_CH1))

        Layer0_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x94140000, End_Address=0x94210000)
        Layer0_1st_Iter_Image5_CH1_256 = data_32_to_16(Layer0_1st_Iter_Image5_CH1)
        #if DEBUG: print("ch1 image 5 : ", len(Layer0_1st_Iter_Image5_CH1))

        Layer0_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x94210000, End_Address=0x942E0000)
        Layer0_1st_Iter_Image6_CH1_256 = data_32_to_16(Layer0_1st_Iter_Image6_CH1)
        #if DEBUG: print("ch1 image 6 : ", len(Layer0_1st_Iter_Image6_CH1))

        Layer0_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x942E0000, End_Address=0x943B0000)
        Layer0_1st_Iter_Image7_CH1_256 = data_32_to_16(Layer0_1st_Iter_Image7_CH1)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Layer0_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x943B0000, End_Address=0x94480000)
        Layer0_1st_Iter_Image8_CH1_256 = data_32_to_16(Layer0_1st_Iter_Image8_CH1)
        #if DEBUG: print("ch1 image 8 : ", len(Layer0_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR & 32bit to 16bit Convert :",e-s)

        '''
        test_out = '1st_iter_result/Layer0_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image1_CH0:
                test_output.write(str(item) + "\n")
        test_output.close()
        
        test_out = '1st_iter_result/Layer0_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer0_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer0_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''

        s = time.time()
        Output_Image1_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image1_CH0_256, Layer0_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image2_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image2_CH0_256, Layer0_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image3_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image3_CH0_256, Layer0_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image4_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image4_CH0_256, Layer0_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image5_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image5_CH0_256, Layer0_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image6_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image6_CH0_256, Layer0_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image7_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image7_CH0_256, Layer0_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Image8_Layer0_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer0_1st_Iter_Image8_CH0_256, Layer0_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)

        OutImages_1st_Layer0 = Output_Image1_Layer0_1st_Iter + Output_Image2_Layer0_1st_Iter + Output_Image3_Layer0_1st_Iter + Output_Image4_Layer0_1st_Iter + \
                            Output_Image5_Layer0_1st_Iter + Output_Image6_Layer0_1st_Iter + Output_Image7_Layer0_1st_Iter + Output_Image8_Layer0_1st_Iter    

        OutImage_1st_Layer0 = torch.tensor([float(value) for value in OutImages_1st_Layer0], dtype=torch.float32).reshape(8, 16, 208, 208)
        
        if DEBUG2 : Save_File(OutImage_1st_Layer0, "result/Layer_0_Forward_1st_Iteration")
        
        if DEBUG: print(OutImage_1st_Layer0[0][0][0][0:5])
        
        # Mean, Var
        s = time.time()
        Mean_1st_Layer0, Var_1st_Layer0 = Cal_mean_var.forward(OutImage_1st_Layer0)    
        e = time.time()
        if DEBUG: print("Calculate Mean & Var :",e-s)

        Beta_Layer0 = data.Beta_Dec[0]
        Gamma_Layer0 = data.Gamma_Dec[0]

        if DEBUG2 : Save_File(data.Beta_Dec[0],   "result/Layer_0_Forward_Beta_Before_Weight_Update")
        if DEBUG2 : Save_File(data.Gamma_Dec[0],  "result/Layer_0_Forward_Gamma_Before_Weight_Update")
        if DEBUG2 : Save_File(data.Weight_Dec[0], "result/Layer_0_Forward_weight_Before_Weight_Update")

        # layer0 Caches: 
        layer0_cache = BN(OutImage_1st_Layer0, Gamma_Layer0, Beta_Layer0)

        # Squeeze to remove the dimension but keeping the same data ordering
        Var_1st_Layer0 = Var_1st_Layer0.squeeze() * Gamma_Layer0
        s = time.time()
        Mean_1st_Layer0, Var_1st_Layer0 = Mean_Var_Dec2Bfloat(Mean_1st_Layer0, Var_1st_Layer0, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat :",e-s)
        s= time.time()
        Weight_2nd_Layer0 = New_Weight_Hardware_ReOrdering_Layer0(16, 16, data.Weight_Bfloat[0], Mean_1st_Layer0, Var_1st_Layer0, data.Beta_Bfloat[0], Iteration="2")
        #if DEBUG: print("Weight_2nd_Layer0 : ", Weight_2nd_Layer0)
        e = time.time()
 
        '''
        data_read_mean_var = "result/Mean_1st_Layer0.txt"
        with open(data_read_mean_var, mode="w") as output_file:  
            for sublist in Mean_1st_Layer0:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n") 
        output_file.close() 

        data_read_mean_var = "result/Var_1st_Layer0.txt"
        with open(data_read_mean_var, mode="w") as output_file:  
            for sublist in Var_1st_Layer0:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n") 
        output_file.close() 
        
        data_read_mean_var = "result/layer0_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:  
            for sublist in Weight_2nd_Layer0:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n") 
        output_file.close()               
        '''
 
        '''
        weight_layer0_2nd_ch0 = data_256_32(Weight_2nd_Layer0[0])
        weight_layer0_2nd_ch1 = data_256_32(Weight_2nd_Layer0[1])
        
        data_read_mean_var = "result/weight_layer0_2nd_ch0.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in weight_layer0_2nd_ch0:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")      
        output_file.close()  

        data_read_mean_var = "result/weight_layer0_2nd_ch1.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in weight_layer0_2nd_ch1:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write
        '''
        s = time.time()
        Write_DDR(data_256_32(Weight_2nd_Layer0[0]), Wr_Address=0x80000000)
        Write_DDR(data_256_32(Weight_2nd_Layer0[1]), Wr_Address=0x90000000)
        e = time.time()
        if DEBUG: print("Write DDR & 256bit to 32bit",e-s)
        
        resume()
        layer0_end = time.time()
        process = layer0_end - layer0_start
        if DEBUG: print("layer0 process time : ", process)

        '''
        d = Device("0000:08:00.0")
        bar = d.bar[0]

        data_read = open("result/layer0_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 
        
        d = Device("0000:08:00.0")
        bar = d.bar[2]
        
        data_read = open("result/layer0_result_ch0_image1.txt", mode="w+")
        i=0
        for i in range(0,int((0X4550000-0X4480000)/4) ): 
            Read_Data = bar.read(0X4480000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer0_result_ch1_image1.txt", mode="w+")
        i=0
        for i in range(0,int((0X4550000-0X4480000)/4) ): 
            Read_Data = bar.read(0X14480000 + (i*4))
            data_read.write(str(Read_Data) + "\n")     

        data_read = open("result/layer0_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X784C000-0X71CC000)/4) ): 
            Read_Data = bar.read(0X71CC000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      
        '''
            
        #################################################
        #                Layer 1 Start                  #
        #################################################
        # check Layer1 IRQ
        check_irq_otherlayer()     
        # self.app_instance .change_color(self.app_instance.L2_IRQ_canvas, self.app_instance.L2_IRQ, "green") 
        # Layer 1
        
        Image0_2nd_ch0 = Read_DDR(Rd_Address=0X84480000, End_Address=0X84550000)
        Image0_2nd_ch0 = data_32_to_16(Image0_2nd_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Image0_2nd_ch1 = Read_DDR(Rd_Address=0X94480000, End_Address=0X94550000)
        Image0_2nd_ch1 = data_32_to_16(Image0_2nd_ch1)
        
        Image0_2nd_result = Read_OutFmap_Bfloat2Dec(Image0_2nd_ch0, Image0_2nd_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False) 

        Image1_2nd_ch0 = Read_DDR(Rd_Address=0X84550000, End_Address=0X84620000)
        Image1_2nd_ch0 = data_32_to_16(Image1_2nd_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Image1_2nd_ch1 = Read_DDR(Rd_Address=0X94550000, End_Address=0X94620000)
        Image1_2nd_ch1 = data_32_to_16(Image1_2nd_ch1)
        
        Image1_2nd_result = Read_OutFmap_Bfloat2Dec(Image1_2nd_ch0, Image1_2nd_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False) 

        Image2_2nd_ch0 = Read_DDR(Rd_Address=0X84620000, End_Address=0X846F0000)
        Image2_2nd_ch0 = data_32_to_16(Image2_2nd_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Image2_2nd_ch1 = Read_DDR(Rd_Address=0X94620000, End_Address=0X946F0000)
        Image2_2nd_ch1 = data_32_to_16(Image2_2nd_ch1)
        
        Image2_2nd_result = Read_OutFmap_Bfloat2Dec(Image2_2nd_ch0, Image2_2nd_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False) 

        Image3_2nd_ch0 = Read_DDR(Rd_Address=0X846F0000, End_Address=0X847C0000)
        Image3_2nd_ch0 = data_32_to_16(Image3_2nd_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Image3_2nd_ch1 = Read_DDR(Rd_Address=0X946F0000, End_Address=0X947C0000)
        Image3_2nd_ch1 = data_32_to_16(Image3_2nd_ch1)
        
        Image3_2nd_result = Read_OutFmap_Bfloat2Dec(Image3_2nd_ch0, Image3_2nd_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False) 

        Image4_2nd_ch0 = Read_DDR(Rd_Address=0X847C0000, End_Address=0X84890000)
        Image4_2nd_ch0 = data_32_to_16(Image4_2nd_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Image4_2nd_ch1 = Read_DDR(Rd_Address=0X947C0000, End_Address=0X94890000)
        Image4_2nd_ch1 = data_32_to_16(Image4_2nd_ch1)
        
        Image4_2nd_result = Read_OutFmap_Bfloat2Dec(Image4_2nd_ch0, Image4_2nd_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False) 
        
        Image5_2nd_ch0 = Read_DDR(Rd_Address=0X84890000, End_Address=0X84960000)
        Image5_2nd_ch0 = data_32_to_16(Image5_2nd_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Image5_2nd_ch1 = Read_DDR(Rd_Address=0X94890000, End_Address=0X94960000)
        Image5_2nd_ch1 = data_32_to_16(Image5_2nd_ch1)
        
        Image5_2nd_result = Read_OutFmap_Bfloat2Dec(Image5_2nd_ch0, Image5_2nd_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False) 

        Image6_2nd_ch0 = Read_DDR(Rd_Address=0X84960000, End_Address=0X84A30000)
        Image6_2nd_ch0 = data_32_to_16(Image6_2nd_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Image6_2nd_ch1 = Read_DDR(Rd_Address=0X94960000, End_Address=0X94A30000)
        Image6_2nd_ch1 = data_32_to_16(Image6_2nd_ch1)
        
        Image6_2nd_result = Read_OutFmap_Bfloat2Dec(Image6_2nd_ch0, Image6_2nd_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False) 

        Image7_2nd_ch0 = Read_DDR(Rd_Address=0X84A30000, End_Address=0X84B00000)
        Image7_2nd_ch0 = data_32_to_16(Image7_2nd_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Image7_2nd_ch1 = Read_DDR(Rd_Address=0X94A30000, End_Address=0X94B00000)
        Image7_2nd_ch1 = data_32_to_16(Image7_2nd_ch1)
        
        Image7_2nd_result = Read_OutFmap_Bfloat2Dec(Image7_2nd_ch0, Image7_2nd_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False) 

        Image_2nd_result = Image0_2nd_result + Image1_2nd_result + Image2_2nd_result + Image3_2nd_result + Image4_2nd_result + Image5_2nd_result +\
                           Image6_2nd_result + Image7_2nd_result
        
        Image_2nd_result = torch.tensor([float(value) for value in Image_2nd_result], dtype=torch.float32).reshape(8, 16, 208, 208)
        
        # Save_File(Image0_2nd_result, "result/Image0_2nd_result")
        # Save_File(Image1_2nd_result, "result/Image1_2nd_result")
        # Save_File(Image2_2nd_result, "result/Image2_2nd_result")
        # Save_File(Image3_2nd_result, "result/Image3_2nd_result")
        # Save_File(Image4_2nd_result, "result/Image4_2nd_result")
        # Save_File(Image5_2nd_result, "result/Image5_2nd_result")
        # Save_File(Image6_2nd_result, "result/Image6_2nd_result")
        # Save_File(Image7_2nd_result, "result/Image7_2nd_result")
        # Save_File(Image_2nd_result, "result/Image_2nd_result")
        
        _Out_2nd_Itr = [Image0_2nd_result, Image1_2nd_result, Image2_2nd_result, Image3_2nd_result, Image4_2nd_result, Image5_2nd_result, Image6_2nd_result, Image7_2nd_result]
        if DEBUG2 : Save_File(_Out_2nd_Itr, "result/Layer_0_Forward_2nd_Iteration") 
        
        
        
        layer1_start = time.time()
        s = time.time()
        # Read DDR & Conver Format # 512MB
        Layer1_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x84B00000, End_Address=0x84CA0000)
        Layer1_1st_Iter_Image1_CH0_256 = (data_32_to_16(Layer1_1st_Iter_Image1_CH0))   
        #if DEBUG: print("ch0 image 1 : ", len(Layer1_1st_Iter_Image1_CH0))    
        
        Layer1_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x84CA0000, End_Address=0x84E40000)
        Layer1_1st_Iter_Image2_CH0_256 = (data_32_to_16(Layer1_1st_Iter_Image2_CH0))
        #if DEBUG: print("ch0 image 2 : ", len(Layer1_1st_Iter_Image2_CH0))
        
        Layer1_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x84E40000, End_Address=0x84FE0000)
        Layer1_1st_Iter_Image3_CH0_256 = (data_32_to_16(Layer1_1st_Iter_Image3_CH0))
        #if DEBUG: print("ch0 image 3 : ", len(Layer1_1st_Iter_Image3_CH0))

        Layer1_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x84FE0000, End_Address=0x85180000)
        Layer1_1st_Iter_Image4_CH0_256 = (data_32_to_16(Layer1_1st_Iter_Image4_CH0))
        #if DEBUG: print("ch0 image 4 : ", len(Layer1_1st_Iter_Image4_CH0))

        Layer1_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x85180000, End_Address=0x85320000)
        Layer1_1st_Iter_Image5_CH0_256 = (data_32_to_16(Layer1_1st_Iter_Image5_CH0))
        #if DEBUG: print("ch0 image 5 : ", len(Layer1_1st_Iter_Image5_CH0))

        Layer1_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x85320000, End_Address=0x854C0000)
        Layer1_1st_Iter_Image6_CH0_256 = (data_32_to_16(Layer1_1st_Iter_Image6_CH0))
        #if DEBUG: print("ch0 image 6 : ", len(Layer1_1st_Iter_Image6_CH0))

        Layer1_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x854C0000, End_Address=0x85660000)
        Layer1_1st_Iter_Image7_CH0_256 = (data_32_to_16(Layer1_1st_Iter_Image7_CH0))
        #if DEBUG: print("ch0 image 7 : ", len(Layer1_1st_Iter_Image7_CH0))

        Layer1_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x85660000, End_Address=0x85800000)
        Layer1_1st_Iter_Image8_CH0_256 = (data_32_to_16(Layer1_1st_Iter_Image8_CH0))
        #if DEBUG: print("ch0 image 8 : ", len(Layer1_1st_Iter_Image8_CH0))


        Layer1_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x94B00000, End_Address=0x94CA0000)
        Layer1_1st_Iter_Image1_CH1_256 = (data_32_to_16(Layer1_1st_Iter_Image1_CH1))
        #if DEBUG: print("ch1 image 1 : ", len(Layer1_1st_Iter_Image1_CH1))

        Layer1_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x94CA0000, End_Address=0x94E40000)
        Layer1_1st_Iter_Image2_CH1_256 = (data_32_to_16(Layer1_1st_Iter_Image2_CH1))
        #if DEBUG: print("ch1 image 2 : ", len(Layer1_1st_Iter_Image2_CH1))

        Layer1_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x94E40000, End_Address=0x94FE0000)
        Layer1_1st_Iter_Image3_CH1_256 = (data_32_to_16(Layer1_1st_Iter_Image3_CH1))
        #if DEBUG: print("ch1 image 3 : ", len(Layer1_1st_Iter_Image3_CH1))

        Layer1_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x94FE0000, End_Address=0x95180000)
        Layer1_1st_Iter_Image4_CH1_256 = (data_32_to_16(Layer1_1st_Iter_Image4_CH1))
        #if DEBUG: print("ch1 image 4 : ", len(Layer1_1st_Iter_Image4_CH1))

        Layer1_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x95180000, End_Address=0x95320000)
        Layer1_1st_Iter_Image5_CH1_256 = (data_32_to_16(Layer1_1st_Iter_Image5_CH1))
        #if DEBUG: print("ch1 image 5 : ", len(Layer1_1st_Iter_Image5_CH1))

        Layer1_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x95320000, End_Address=0x954C0000)
        Layer1_1st_Iter_Image6_CH1_256 = (data_32_to_16(Layer1_1st_Iter_Image6_CH1))
        #if DEBUG: print("ch1 image 6 : ", len(Layer1_1st_Iter_Image6_CH1))

        Layer1_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x954C0000, End_Address=0x95660000)
        Layer1_1st_Iter_Image7_CH1_256 = (data_32_to_16(Layer1_1st_Iter_Image7_CH1))
        #if DEBUG: print("ch1 image 7 : ", len(Layer1_1st_Iter_Image7_CH1))

        Layer1_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x95660000, End_Address=0x95800000)
        Layer1_1st_Iter_Image8_CH1_256 = (data_32_to_16(Layer1_1st_Iter_Image8_CH1))
        #if DEBUG: print("ch1 image 8 : ", len(Layer1_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR & 32bit to 16bit :",e-s)

        '''
        test_out = '1st_iter_result/Layer1_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer1_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer1_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''

        s = time.time()
        Output_Image1_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image1_CH0_256, Layer1_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image2_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image2_CH0_256, Layer1_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image3_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image3_CH0_256, Layer1_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image4_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image4_CH0_256, Layer1_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image5_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image5_CH0_256, Layer1_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image6_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image6_CH0_256, Layer1_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image7_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image7_CH0_256, Layer1_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        Output_Image8_Layer1_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer1_1st_Iter_Image8_CH0_256, Layer1_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=208, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec :",e-s)
        
        OutImages_1st_Layer1 = Output_Image1_Layer1_1st_Iter + Output_Image2_Layer1_1st_Iter + Output_Image3_Layer1_1st_Iter + Output_Image4_Layer1_1st_Iter + \
                            Output_Image5_Layer1_1st_Iter + Output_Image6_Layer1_1st_Iter + Output_Image7_Layer1_1st_Iter + Output_Image8_Layer1_1st_Iter    

        OutImage_1st_Layer1 = torch.tensor([float(value) for value in OutImages_1st_Layer1], dtype=torch.float32).reshape(8, 32, 208, 208)
        if DEBUG: print(OutImage_1st_Layer1[0][0][0][0:5])
        
        if DEBUG2 : Save_File(data.Beta_Dec[1],   "result/Layer_1_Forward_Beta_Before_Weight_Update")
        if DEBUG2 : Save_File(data.Gamma_Dec[1],  "result/Layer_1_Forward_Gamma_Before_Weight_Update")
        if DEBUG2 : Save_File(data.Weight_Dec[1], "result/Layer_1_Forward_weight_Before_Weight_Update")
        
        # Mean, Var
        s = time.time()
        Mean_1st_Layer1, Var_1st_Layer1 = Cal_mean_var.forward(OutImage_1st_Layer1)
        e = time.time()
        if DEBUG: print("Cacluate Mean & Var :",e-s)
        
        Beta_Layer1 = data.Beta_Dec[1]
        Gamma_Layer1 = data.Gamma_Dec[1]

        layer1_cache = BN(OutImage_1st_Layer1, Gamma_Layer1, Beta_Layer1)

        # Squeeze to remove the dimension but keeping the same data ordering
        Var_1st_Layer1 = Var_1st_Layer1.squeeze() * Gamma_Layer1

        s = time.time()
        Mean_1st_Layer1, Var_1st_Layer1 = Mean_Var_Dec2Bfloat(Mean_1st_Layer1, Var_1st_Layer1, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat :",e-s)
        s = time.time()
        Weight_2nd_Layer1 = New_Weight_Hardware_ReOrdering_OtherLayer(32, 16, data.Weight_Bfloat[1], Mean_1st_Layer1, Var_1st_Layer1, data.Beta_Bfloat[1], Iteration="2")
        e = time.time()
        if DEBUG: print("Weight Reordering :",e-s)

        # Write DDR
        s = time.time()
        Write_DDR(data_256_32(Weight_2nd_Layer1[0]), Wr_Address=0x80000A00)
        Write_DDR(data_256_32(Weight_2nd_Layer1[1]), Wr_Address=0x90000A00)
        e = time.time()
        if DEBUG: print("Write DDR & 256bit to 32 bit :",e-s)

        layer1_end = time.time()
        layer1_process = layer1_end - layer1_start
        if DEBUG: print("Layer1 process time : ", layer1_process)

        resume()

        '''
        d = Device("0000:08:00.0")
        bar = d.bar[0]

        data_read = open("result/layer1_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        d = Device("0000:08:00.0")
        bar = d.bar[2]

        data_read = open("result/layer1_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X5B40000-0X5800000)/4) ): 
            Read_Data = bar.read(0X5800000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer1_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X5B40000-0X5800000)/4) ): 
            Read_Data = bar.read(0X15800000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer1_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X7B8C000-0X784C000)/4) ): 
            Read_Data = bar.read(0X784C000 + (i*4))
            data_read.write(str(Read_Data) + "\n")     
        '''   

        #################################################
        #                Layer 2 Start                  #
        #################################################
        # check Layer2 IRQ
        check_irq_otherlayer()
        # self.app_instance .change_color(self.app_instance.L3_IRQ_canvas, self.app_instance.L3_IRQ, "green")
        # Layer 2
        layer2_start = time.time()
        # Read DDR & Conver Format # 512MB
        s = time.time()
        Layer2_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x85B40000, End_Address=0x85C10000)
        Layer2_1st_Iter_Image1_CH0_256 = (data_32_to_16(Layer2_1st_Iter_Image1_CH0))   
        #if DEBUG: print("ch0 image 1 : ", len(Layer2_1st_Iter_Image1_CH0))     

        Layer2_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x85C10000, End_Address=0x85CE0000)
        Layer2_1st_Iter_Image2_CH0_256 = (data_32_to_16(Layer2_1st_Iter_Image2_CH0))
        #if DEBUG: print("ch0 image 2 : ", len(Layer2_1st_Iter_Image2_CH0))
        
        Layer2_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x85CE0000, End_Address=0x85DB0000)
        Layer2_1st_Iter_Image3_CH0_256 = (data_32_to_16(Layer2_1st_Iter_Image3_CH0))
        #if DEBUG: print("ch0 image 3 : ", len(Layer2_1st_Iter_Image3_CH0))

        Layer2_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x85DB0000, End_Address=0x85E80000)
        Layer2_1st_Iter_Image4_CH0_256 = (data_32_to_16(Layer2_1st_Iter_Image4_CH0))
        #if DEBUG: print("ch0 image 4 : ", len(Layer2_1st_Iter_Image4_CH0))

        Layer2_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x85E80000, End_Address=0x85F50000)
        Layer2_1st_Iter_Image5_CH0_256 = (data_32_to_16(Layer2_1st_Iter_Image5_CH0))
        #if DEBUG: print("ch0 image 5 : ", len(Layer2_1st_Iter_Image5_CH0))

        Layer2_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x85F50000, End_Address=0x86020000)
        Layer2_1st_Iter_Image6_CH0_256 = (data_32_to_16(Layer2_1st_Iter_Image6_CH0))
        #if DEBUG: print("ch0 image 6 : ", len(Layer2_1st_Iter_Image6_CH0))

        Layer2_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x86020000, End_Address=0x860F0000)
        Layer2_1st_Iter_Image7_CH0_256 = (data_32_to_16(Layer2_1st_Iter_Image7_CH0))
        #if DEBUG: print("ch0 image 7 : ", len(Layer2_1st_Iter_Image7_CH0))

        Layer2_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x860F0000, End_Address=0x861C0000)
        Layer2_1st_Iter_Image8_CH0_256 = (data_32_to_16(Layer2_1st_Iter_Image8_CH0))
        #if DEBUG: print("ch0 image 8 : ", len(Layer2_1st_Iter_Image8_CH0))


        Layer2_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x95B40000, End_Address=0x95C10000)
        Layer2_1st_Iter_Image1_CH1_256 = (data_32_to_16(Layer2_1st_Iter_Image1_CH1))
        #if DEBUG: print("ch1 image 1 : ", len(Layer2_1st_Iter_Image1_CH1))

        Layer2_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x95C10000, End_Address=0x95CE0000)
        Layer2_1st_Iter_Image2_CH1_256 = (data_32_to_16(Layer2_1st_Iter_Image2_CH1))
        #if DEBUG: print("ch1 image 2 : ", len(Layer2_1st_Iter_Image2_CH1))

        Layer2_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x95CE0000, End_Address=0x95DB0000)
        Layer2_1st_Iter_Image3_CH1_256 = (data_32_to_16(Layer2_1st_Iter_Image3_CH1))
        #if DEBUG: print("ch1 image 3 : ", len(Layer2_1st_Iter_Image3_CH1))

        Layer2_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x95DB0000, End_Address=0x95E80000)
        Layer2_1st_Iter_Image4_CH1_256 = (data_32_to_16(Layer2_1st_Iter_Image4_CH1))
        #if DEBUG: print("ch1 image 4 : ", len(Layer2_1st_Iter_Image4_CH1))

        Layer2_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x95E80000, End_Address=0x95F50000)
        Layer2_1st_Iter_Image5_CH1_256 = (data_32_to_16(Layer2_1st_Iter_Image5_CH1))
        #if DEBUG: print("ch1 image 5 : ", len(Layer2_1st_Iter_Image5_CH1))

        Layer2_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x95F50000, End_Address=0x96020000)
        Layer2_1st_Iter_Image6_CH1_256 = (data_32_to_16(Layer2_1st_Iter_Image6_CH1))
        #if DEBUG: print("ch1 image 6 : ", len(Layer2_1st_Iter_Image6_CH1))

        Layer2_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x96020000, End_Address=0x960F0000)
        Layer2_1st_Iter_Image7_CH1_256 = (data_32_to_16(Layer2_1st_Iter_Image7_CH1))
        #if DEBUG: print("ch1 image 7 : ", len(Layer2_1st_Iter_Image7_CH1))

        Layer2_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x960F0000, End_Address=0x961C0000)
        Layer2_1st_Iter_Image8_CH1_256 = (data_32_to_16(Layer2_1st_Iter_Image8_CH1))
        #if DEBUG: print("ch1 image 8 : ", len(Layer2_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR & 32bit to 16bit :",e-s)

        '''
        test_out = '1st_iter_result/Layer2_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer2_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer2_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''

        s = time.time()
        Output_Image1_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image1_CH0_256, Layer2_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image2_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image2_CH0_256, Layer2_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image3_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image3_CH0_256, Layer2_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image4_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image4_CH0_256, Layer2_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image5_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image5_CH0_256, Layer2_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image6_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image6_CH0_256, Layer2_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image7_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image7_CH0_256, Layer2_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        Output_Image8_Layer2_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer2_1st_Iter_Image8_CH0_256, Layer2_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=104, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec :",e-s)

        OutImages_1st_Layer2 = Output_Image1_Layer2_1st_Iter + Output_Image2_Layer2_1st_Iter + Output_Image3_Layer2_1st_Iter + Output_Image4_Layer2_1st_Iter + \
                            Output_Image5_Layer2_1st_Iter + Output_Image6_Layer2_1st_Iter + Output_Image7_Layer2_1st_Iter + Output_Image8_Layer2_1st_Iter    

        OutImage_1st_Layer2 = torch.tensor([float(value) for value in OutImages_1st_Layer2], dtype=torch.float32).reshape(8, 64, 104, 104)
        # Mean, Var
        s = time.time()
        Mean_1st_Layer2, Var_1st_Layer2 = Cal_mean_var.forward(OutImage_1st_Layer2)
        e = time.time()
        if DEBUG: print("Calcuulate Mean & Var :",e-s)

        Beta_Layer2 = data.Beta_Dec[2]
        Gamma_Layer2 = data.Gamma_Dec[2]

        layer2_cache = BN(OutImage_1st_Layer2, Gamma_Layer2, Beta_Layer2)

        # Squeeze to remove the dimension but keeping the same data ordering
        Var_1st_Layer2 = Var_1st_Layer2.squeeze() * Gamma_Layer2

        s = time.time()
        Mean_1st_Layer2, Var_1st_Layer2 = Mean_Var_Dec2Bfloat(Mean_1st_Layer2, Var_1st_Layer2, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat :",e-s)
        s = time.time()
        Weight_2nd_Layer2 = New_Weight_Hardware_ReOrdering_OtherLayer(64, 32, data.Weight_Bfloat[2], Mean_1st_Layer2, Var_1st_Layer2, data.Beta_Bfloat[2], Iteration="2")
        e = time.time()
        if DEBUG: print("Weight Reordering :",e-s)

        '''
        data_read_mean_var = "result/layer2_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer2:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")     
        output_file.close()    
        '''       

        # Write DDR
        s = time.time()
        Write_DDR(data_256_32(Weight_2nd_Layer2[0]), Wr_Address=0x80001E00)
        Write_DDR(data_256_32(Weight_2nd_Layer2[1]), Wr_Address=0x90001E00)
        e = time.time()
        if DEBUG: print("Write DDR & 256bit to 32bit :",e-s)
        
        layer2_end = time.time()
        layer2_process = layer2_end - layer2_start
        if DEBUG: print("Layer2 process time : ", layer2_process)

        resume()

        '''
        d = Device("0000:08:00.0")
        bar = d.bar[0]

        data_read = open("result/layer_2_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        d = Device("0000:08:00.0")
        bar = d.bar[2]

        data_read = open("result/layer2_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X6360000-0X61C0000)/4) ): 
            Read_Data = bar.read(0X61C0000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer2_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X6360000-0X61C0000)/4) ): 
            Read_Data = bar.read(0X161C0000 + (i*4))
            data_read.write(str(Read_Data) + "\n")

        data_read = open("result/layer2_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X7D2C000-0X7B8C000)/4) ): 
            Read_Data = bar.read(0X7B8C000 + (i*4))
            data_read.write(str(Read_Data) + "\n")     
        '''


        #################################################
        #                Layer 3 Start                  #
        #################################################
        # check Layer3 IRQ
        check_irq_otherlayer()
        # self.app_instance .change_color(self.app_instance.L4_IRQ_canvas, self.app_instance.L4_IRQ, "green")
        # Layer 3
        layer3_start = time.time()
        # Read DDR & Conver Format # 512MB
        s = time.time()
        Layer3_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x86360000, End_Address=0x863C8000)
        Layer3_1st_Iter_Image1_CH0_256 = (data_32_to_16(Layer3_1st_Iter_Image1_CH0))   
        #if DEBUG: print("ch0 image 1 : ", len(Layer3_1st_Iter_Image1_CH0))     

        Layer3_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x863C8000, End_Address=0x86430000)
        Layer3_1st_Iter_Image2_CH0_256 = (data_32_to_16(Layer3_1st_Iter_Image2_CH0))
        #if DEBUG: print("ch0 image 2 : ", len(Layer3_1st_Iter_Image2_CH0))
        
        Layer3_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x86430000, End_Address=0x86498000)
        Layer3_1st_Iter_Image3_CH0_256 = (data_32_to_16(Layer3_1st_Iter_Image3_CH0))
        #if DEBUG: print("ch0 image 3 : ", len(Layer3_1st_Iter_Image3_CH0))

        Layer3_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x86498000, End_Address=0x86500000)
        Layer3_1st_Iter_Image4_CH0_256 = (data_32_to_16(Layer3_1st_Iter_Image4_CH0))
        #if DEBUG: print("ch0 image 4 : ", len(Layer3_1st_Iter_Image4_CH0))

        Layer3_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x86500000, End_Address=0x86568000)
        Layer3_1st_Iter_Image5_CH0_256 = (data_32_to_16(Layer3_1st_Iter_Image5_CH0))
        #if DEBUG: print("ch0 image 5 : ", len(Layer3_1st_Iter_Image5_CH0))

        Layer3_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x86568000, End_Address=0x865D0000)
        Layer3_1st_Iter_Image6_CH0_256 = (data_32_to_16(Layer3_1st_Iter_Image6_CH0))
        #if DEBUG: print("ch0 image 6 : ", len(Layer3_1st_Iter_Image6_CH0))

        Layer3_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x865D0000, End_Address=0x86638000)
        Layer3_1st_Iter_Image7_CH0_256 = (data_32_to_16(Layer3_1st_Iter_Image7_CH0))
        #if DEBUG: print("ch0 image 7 : ", len(Layer3_1st_Iter_Image7_CH0))

        Layer3_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x86638000, End_Address=0x866A0000)
        Layer3_1st_Iter_Image8_CH0_256 = (data_32_to_16(Layer3_1st_Iter_Image8_CH0))
        #if DEBUG: print("ch0 image 8 : ", len(Layer3_1st_Iter_Image8_CH0))


        Layer3_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x96360000, End_Address=0x963C8000)
        Layer3_1st_Iter_Image1_CH1_256 = (data_32_to_16(Layer3_1st_Iter_Image1_CH1))
        #if DEBUG: print("ch1 image 1 : ", len(Layer3_1st_Iter_Image1_CH1))

        Layer3_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x963C8000, End_Address=0x96430000)
        Layer3_1st_Iter_Image2_CH1_256 = (data_32_to_16(Layer3_1st_Iter_Image2_CH1))
        #if DEBUG: print("ch1 image 2 : ", len(Layer3_1st_Iter_Image2_CH1))

        Layer3_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x96430000, End_Address=0x96498000)
        Layer3_1st_Iter_Image3_CH1_256 = (data_32_to_16(Layer3_1st_Iter_Image3_CH1))
        #if DEBUG: print("ch1 image 3 : ", len(Layer3_1st_Iter_Image3_CH1))

        Layer3_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x96498000, End_Address=0x96500000)
        Layer3_1st_Iter_Image4_CH1_256 = (data_32_to_16(Layer3_1st_Iter_Image4_CH1))
        #if DEBUG: print("ch1 image 4 : ", len(Layer3_1st_Iter_Image4_CH1))

        Layer3_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x96500000, End_Address=0x96568000)
        Layer3_1st_Iter_Image5_CH1_256 = (data_32_to_16(Layer3_1st_Iter_Image5_CH1))
        #if DEBUG: print("ch1 image 5 : ", len(Layer3_1st_Iter_Image5_CH1))

        Layer3_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x96568000, End_Address=0x965D0000)
        Layer3_1st_Iter_Image6_CH1_256 = (data_32_to_16(Layer3_1st_Iter_Image6_CH1))
        #if DEBUG: print("ch1 image 6 : ", len(Layer3_1st_Iter_Image6_CH1))

        Layer3_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x965D0000, End_Address=0x96638000)
        Layer3_1st_Iter_Image7_CH1_256 = (data_32_to_16(Layer3_1st_Iter_Image7_CH1))
        #if DEBUG: print("ch1 image 7 : ", len(Layer3_1st_Iter_Image7_CH1))

        Layer3_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x96638000, End_Address=0x966A0000)
        Layer3_1st_Iter_Image8_CH1_256 = (data_32_to_16(Layer3_1st_Iter_Image8_CH1))
        #if DEBUG: print("ch1 image 8 : ", len(Layer3_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR & 32bit to 16bit : ",e-s)

        '''
        test_out = '1st_iter_result/Layer3_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer3_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer3_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''
        
        s = time.time()
        Output_Image1_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image1_CH0_256, Layer3_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image2_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image2_CH0_256, Layer3_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image3_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image3_CH0_256, Layer3_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image4_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image4_CH0_256, Layer3_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image5_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image5_CH0_256, Layer3_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image6_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image6_CH0_256, Layer3_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image7_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image7_CH0_256, Layer3_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        Output_Image8_Layer3_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer3_1st_Iter_Image8_CH0_256, Layer3_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=52, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)

        OutImages_1st_Layer3 = Output_Image1_Layer3_1st_Iter + Output_Image2_Layer3_1st_Iter + Output_Image3_Layer3_1st_Iter + Output_Image4_Layer3_1st_Iter + \
                            Output_Image5_Layer3_1st_Iter + Output_Image6_Layer3_1st_Iter + Output_Image7_Layer3_1st_Iter + Output_Image8_Layer3_1st_Iter    

        OutImage_1st_Layer3 = torch.tensor([float(value) for value in OutImages_1st_Layer3], dtype=torch.float32).reshape(8, 128, 52, 52)

        # Mean, Var
        Mean_1st_Layer3, Var_1st_Layer3 = Cal_mean_var.forward(OutImage_1st_Layer3)


        Beta_Layer3 = data.Beta_Dec[3]
        Gamma_Layer3 = data.Gamma_Dec[3]

        layer3_cache = BN(OutImage_1st_Layer3, Gamma_Layer3, Beta_Layer3)

        # Squeeze to remove the dimension but keeping the same data ordering
        Var_1st_Layer3 = Var_1st_Layer3.squeeze() * Gamma_Layer3
        s = time.time()
        Mean_1st_Layer3, Var_1st_Layer3 = Mean_Var_Dec2Bfloat(Mean_1st_Layer3, Var_1st_Layer3, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat : ",e-s)
        s = time.time()
        Weight_2nd_Layer3 = New_Weight_Hardware_ReOrdering_OtherLayer(128, 64, data.Weight_Bfloat[3], Mean_1st_Layer3, Var_1st_Layer3, data.Beta_Bfloat[3], Iteration="2")
        e = time.time()
        if DEBUG: print("Weight Reordering : ",e-s)    

        '''
        data_read_mean_var = "result/layer3_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer3:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")     
        '''           

        # Write DDR
        s = time.time()
        Write_DDR(data_256_32(Weight_2nd_Layer3[0]), Wr_Address=0x80006E00)
        Write_DDR(data_256_32(Weight_2nd_Layer3[1]), Wr_Address=0x90006E00)
        e = time.time()
        if DEBUG: print("Write DDR & 256bit to 32bit : ",e-s)

        layer3_end = time.time()
        layer3_process = layer3_end - layer3_start
        if DEBUG: print("Layer3 process time : ", layer3_process)
        
        resume()
        #if DEBUG: print(irq_val)

        '''
        d = Device("0000:08:00.0")
        bar = d.bar[0]

        data_read = open("result/layer3_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        d = Device("0000:08:00.0")
        bar = d.bar[2]

        data_read = open("result/layer3_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X6770000-0X66A0000)/4) ): 
            Read_Data = bar.read(0X66A0000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer3_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X6770000-0X66A0000)/4) ): 
            Read_Data = bar.read(0X166A0000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer3_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X7DFC000-0X7D2C000)/4) ): 
            Read_Data = bar.read(0X7D2C000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      
        '''

        #################################################
        #                Layer 4 Start                  #
        #################################################
        # check Layer4 IRQ
        check_irq_otherlayer()
        # self.app_instance .change_color(self.app_instance.L5_IRQ_canvas, self.app_instance.L5_IRQ, "green")
        # Layer 4
        Layer4_start = time.time()
        # Read DDR & Conver Format # 512MB
        s = time.time()
        Layer4_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x86770000, End_Address=0x867A4000)
        Layer4_1st_Iter_Image1_CH0_256 = (data_32_to_16(Layer4_1st_Iter_Image1_CH0))   
        #if DEBUG: print("ch0 image 1 : ", len(Layer4_1st_Iter_Image1_CH0))     

        Layer4_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x867A4000, End_Address=0x867D8000)
        Layer4_1st_Iter_Image2_CH0_256 = (data_32_to_16(Layer4_1st_Iter_Image2_CH0))
        #if DEBUG: print("ch0 image 2 : ", len(Layer4_1st_Iter_Image2_CH0))
        
        Layer4_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x867D8000, End_Address=0x8680C000)
        Layer4_1st_Iter_Image3_CH0_256 = (data_32_to_16(Layer4_1st_Iter_Image3_CH0))
        #if DEBUG: print("ch0 image 3 : ", len(Layer4_1st_Iter_Image3_CH0))

        Layer4_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x8680C000, End_Address=0x86840000)
        Layer4_1st_Iter_Image4_CH0_256 = (data_32_to_16(Layer4_1st_Iter_Image4_CH0))
        #if DEBUG: print("ch0 image 4 : ", len(Layer4_1st_Iter_Image4_CH0))

        Layer4_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x86840000, End_Address=0x86874000)
        Layer4_1st_Iter_Image5_CH0_256 = (data_32_to_16(Layer4_1st_Iter_Image5_CH0))
        #if DEBUG: print("ch0 image 5 : ", len(Layer4_1st_Iter_Image5_CH0))

        Layer4_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x86874000, End_Address=0x868A8000)
        Layer4_1st_Iter_Image6_CH0_256 = (data_32_to_16(Layer4_1st_Iter_Image6_CH0))
        #if DEBUG: print("ch0 image 6 : ", len(Layer4_1st_Iter_Image6_CH0))

        Layer4_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x868A8000, End_Address=0x868DC000)
        Layer4_1st_Iter_Image7_CH0_256 = (data_32_to_16(Layer4_1st_Iter_Image7_CH0))
        #if DEBUG: print("ch0 image 7 : ", len(Layer4_1st_Iter_Image7_CH0))

        Layer4_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x868DC000, End_Address=0x86910000)
        Layer4_1st_Iter_Image8_CH0_256 = (data_32_to_16(Layer4_1st_Iter_Image8_CH0))
        #if DEBUG: print("ch0 image 8 : ", len(Layer4_1st_Iter_Image8_CH0))


        Layer4_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x96770000, End_Address=0x967A4000)
        Layer4_1st_Iter_Image1_CH1_256 = (data_32_to_16(Layer4_1st_Iter_Image1_CH1))
        #if DEBUG: print("ch1 image 1 : ", len(Layer4_1st_Iter_Image1_CH1))

        Layer4_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x967A4000, End_Address=0x967D8000)
        Layer4_1st_Iter_Image2_CH1_256 = (data_32_to_16(Layer4_1st_Iter_Image2_CH1))
        #if DEBUG: print("ch1 image 2 : ", len(Layer4_1st_Iter_Image2_CH1))

        Layer4_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x967D8000, End_Address=0x9680C000)
        Layer4_1st_Iter_Image3_CH1_256 = (data_32_to_16(Layer4_1st_Iter_Image3_CH1))
        #if DEBUG: print("ch1 image 3 : ", len(Layer4_1st_Iter_Image3_CH1))

        Layer4_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x9680C000, End_Address=0x96840000)
        Layer4_1st_Iter_Image4_CH1_256 = (data_32_to_16(Layer4_1st_Iter_Image4_CH1))
        #if DEBUG: print("ch1 image 4 : ", len(Layer4_1st_Iter_Image4_CH1))

        Layer4_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x96840000, End_Address=0x96874000)
        Layer4_1st_Iter_Image5_CH1_256 = (data_32_to_16(Layer4_1st_Iter_Image5_CH1))
        #if DEBUG: print("ch1 image 5 : ", len(Layer4_1st_Iter_Image5_CH1))

        Layer4_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x96874000, End_Address=0x968A8000)
        Layer4_1st_Iter_Image6_CH1_256 = (data_32_to_16(Layer4_1st_Iter_Image6_CH1))
        #if DEBUG: print("ch1 image 6 : ", len(Layer4_1st_Iter_Image6_CH1))

        Layer4_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x968A8000, End_Address=0x968DC000)
        Layer4_1st_Iter_Image7_CH1_256 = (data_32_to_16(Layer4_1st_Iter_Image7_CH1))
        #if DEBUG: print("ch1 image 7 : ", len(Layer4_1st_Iter_Image7_CH1))

        Layer4_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x968DC000, End_Address=0x96910000)
        Layer4_1st_Iter_Image8_CH1_256 = (data_32_to_16(Layer4_1st_Iter_Image8_CH1))
        #if DEBUG: print("ch1 image 8 : ", len(Layer4_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR & 32bit to 16bit : ",e-s)


        '''
        test_out = '1st_iter_result/Layer4_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer4_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer4_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''
        
        s = time.time()
        Output_Image1_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image1_CH0_256, Layer4_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image2_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image2_CH0_256, Layer4_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image3_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image3_CH0_256, Layer4_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image4_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image4_CH0_256, Layer4_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image5_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image5_CH0_256, Layer4_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image6_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image6_CH0_256, Layer4_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image7_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image7_CH0_256, Layer4_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        Output_Image8_Layer4_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer4_1st_Iter_Image8_CH0_256, Layer4_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=26, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)

        OutImages_1st_Layer4 = Output_Image1_Layer4_1st_Iter + Output_Image2_Layer4_1st_Iter + Output_Image3_Layer4_1st_Iter + Output_Image4_Layer4_1st_Iter + \
                            Output_Image5_Layer4_1st_Iter + Output_Image6_Layer4_1st_Iter + Output_Image7_Layer4_1st_Iter + Output_Image8_Layer4_1st_Iter    

        OutImage_1st_Layer4 = torch.tensor([float(value) for value in OutImages_1st_Layer4], dtype=torch.float32).reshape(8, 256, 26, 26)

        # Mean, Var
        s = time.time()
        Mean_1st_Layer4, Var_1st_Layer4 = Cal_mean_var.forward(OutImage_1st_Layer4)
        e = time.time()
        if DEBUG: print("Calculate Mean & Var : ",e-s)

        Beta_Layer4 = data.Beta_Dec[4]
        Gamma_Layer4 = data.Gamma_Dec[4]

        layer4_cache = BN(OutImage_1st_Layer4, Gamma_Layer4, Beta_Layer4)

        # Squeeze to remove the dimension but keeping the same data ordering
        Var_1st_Layer4 = Var_1st_Layer4.squeeze() * Gamma_Layer4

        s = time.time()
        Mean_1st_Layer4, Var_1st_Layer4 = Mean_Var_Dec2Bfloat(Mean_1st_Layer4, Var_1st_Layer4, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat : ",e-s)
        s = time.time()
        Weight_2nd_Layer4 = New_Weight_Hardware_ReOrdering_OtherLayer(256, 128, data.Weight_Bfloat[4], Mean_1st_Layer4, Var_1st_Layer4, data.Beta_Bfloat[4], Iteration="2")
        e = time.time()
        if DEBUG: print("Weight Reordering : ",e-s)

        '''
        data_read_mean_var = "result/layer4_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer4:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")        
        '''

        # Write DDR
        s = time.time()
        Write_DDR(data_256_32(Weight_2nd_Layer4[0]), Wr_Address=0x8001AE00)
        Write_DDR(data_256_32(Weight_2nd_Layer4[1]), Wr_Address=0x9001AE00)
        e = time.time()
        if DEBUG: print("Write DDR & 256bit to 32bit : ",e-s)

        layer4_end = time.time()
        layer4_process = layer4_end - Layer4_start
        if DEBUG: print("Layer4 process time : ", layer4_process)
        
        resume()

        '''
        d = Device("0000:08:00.0")
        bar = d.bar[0]

        data_read = open("result/layer4_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        d = Device("0000:08:00.0")
        bar = d.bar[2]

        data_read = open("result/layer4_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X6978000-0X6910000)/4) ): 
            Read_Data = bar.read(0X6910000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer4_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X6978000-0X6910000)/4) ): 
            Read_Data = bar.read(0X16910000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer4_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X7E64000-0X7DFC000)/4) ): 
            Read_Data = bar.read(0X7DFC000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      
        '''

        #################################################
        #                Layer 5 Start                  #
        #################################################
        # check Layer5 IRQ
        check_irq_otherlayer()
        # self.app_instance .change_color(self.app_instance.L6_IRQ_canvas, self.app_instance.L6_IRQ, "green")
        # Layer 5
        Layer5_start = time.time()
        s = time.time()
        # Read DDR & Conver Format # 512MB
        Layer5_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x86978000, End_Address=0x86992000)
        Layer5_1st_Iter_Image1_CH0_256 = (data_32_to_16(Layer5_1st_Iter_Image1_CH0))   
        #if DEBUG: print("ch0 image 1 : ", len(Layer5_1st_Iter_Image1_CH0))     

        Layer5_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x86992000, End_Address=0x869AC000)
        Layer5_1st_Iter_Image2_CH0_256 = (data_32_to_16(Layer5_1st_Iter_Image2_CH0))
        #if DEBUG: print("ch0 image 2 : ", len(Layer5_1st_Iter_Image2_CH0))
        
        Layer5_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x869AC000, End_Address=0x869C6000)
        Layer5_1st_Iter_Image3_CH0_256 = (data_32_to_16(Layer5_1st_Iter_Image3_CH0))
        #if DEBUG: print("ch0 image 3 : ", len(Layer5_1st_Iter_Image3_CH0))

        Layer5_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x869C6000, End_Address=0x869E0000)
        Layer5_1st_Iter_Image4_CH0_256 = (data_32_to_16(Layer5_1st_Iter_Image4_CH0))
        #if DEBUG: print("ch0 image 4 : ", len(Layer5_1st_Iter_Image4_CH0))

        Layer5_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x869E0000, End_Address=0x869FA000)
        Layer5_1st_Iter_Image5_CH0_256 = (data_32_to_16(Layer5_1st_Iter_Image5_CH0))
        #if DEBUG: print("ch0 image 5 : ", len(Layer5_1st_Iter_Image5_CH0))

        Layer5_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x869FA000, End_Address=0x86A14000)
        Layer5_1st_Iter_Image6_CH0_256 = (data_32_to_16(Layer5_1st_Iter_Image6_CH0))
        #if DEBUG: print("ch0 image 6 : ", len(Layer5_1st_Iter_Image6_CH0))

        Layer5_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x86A14000, End_Address=0x86A2E000)
        Layer5_1st_Iter_Image7_CH0_256 = (data_32_to_16(Layer5_1st_Iter_Image7_CH0))
        #if DEBUG: print("ch0 image 7 : ", len(Layer5_1st_Iter_Image7_CH0))

        Layer5_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x86A2E000, End_Address=0x86A48000)
        Layer5_1st_Iter_Image8_CH0_256 = (data_32_to_16(Layer5_1st_Iter_Image8_CH0))
        #if DEBUG: print("ch0 image 8 : ", len(Layer5_1st_Iter_Image8_CH0))


        Layer5_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x96978000, End_Address=0x96992000)
        Layer5_1st_Iter_Image1_CH1_256 = (data_32_to_16(Layer5_1st_Iter_Image1_CH1))
        #if DEBUG: print("ch1 image 1 : ", len(Layer5_1st_Iter_Image1_CH1))

        Layer5_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x96992000, End_Address=0x969AC000)
        Layer5_1st_Iter_Image2_CH1_256 = (data_32_to_16(Layer5_1st_Iter_Image2_CH1))
        #if DEBUG: print("ch1 image 2 : ", len(Layer5_1st_Iter_Image2_CH1))

        Layer5_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x969AC000, End_Address=0x969C6000)
        Layer5_1st_Iter_Image3_CH1_256 = (data_32_to_16(Layer5_1st_Iter_Image3_CH1))
        #if DEBUG: print("ch1 image 3 : ", len(Layer5_1st_Iter_Image3_CH1))

        Layer5_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x969C6000, End_Address=0x969E0000)
        Layer5_1st_Iter_Image4_CH1_256 = (data_32_to_16(Layer5_1st_Iter_Image4_CH1))
        #if DEBUG: print("ch1 image 4 : ", len(Layer5_1st_Iter_Image4_CH1))

        Layer5_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x969E0000, End_Address=0x969FA000)
        Layer5_1st_Iter_Image5_CH1_256 = (data_32_to_16(Layer5_1st_Iter_Image5_CH1))
        #if DEBUG: print("ch1 image 5 : ", len(Layer5_1st_Iter_Image5_CH1))

        Layer5_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x969FA000, End_Address=0x96A14000)
        Layer5_1st_Iter_Image6_CH1_256 = (data_32_to_16(Layer5_1st_Iter_Image6_CH1))
        #if DEBUG: print("ch1 image 6 : ", len(Layer5_1st_Iter_Image6_CH1))

        Layer5_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x96A14000, End_Address=0x96A2E000)
        Layer5_1st_Iter_Image7_CH1_256 = (data_32_to_16(Layer5_1st_Iter_Image7_CH1))
        #if DEBUG: print("ch1 image 7 : ", len(Layer5_1st_Iter_Image7_CH1))

        Layer5_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x96A2E000, End_Address=0x96A48000)
        Layer5_1st_Iter_Image8_CH1_256 = (data_32_to_16(Layer5_1st_Iter_Image8_CH1))
        #if DEBUG: print("ch1 image 8 : ", len(Layer5_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR & 32bit to 16bit : ",e-s)

        '''
        test_out = '1st_iter_result/Layer5_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer5_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer5_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''
        

        s = time.time()
        Output_Image1_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image1_CH0_256, Layer5_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image2_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image2_CH0_256, Layer5_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image3_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image3_CH0_256, Layer5_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image4_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image4_CH0_256, Layer5_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image5_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image5_CH0_256, Layer5_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image6_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image6_CH0_256, Layer5_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image7_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image7_CH0_256, Layer5_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Image8_Layer5_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer5_1st_Iter_Image8_CH0_256, Layer5_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)

        OutImages_1st_Layer5 = Output_Image1_Layer5_1st_Iter + Output_Image2_Layer5_1st_Iter + Output_Image3_Layer5_1st_Iter + Output_Image4_Layer5_1st_Iter + \
                            Output_Image5_Layer5_1st_Iter + Output_Image6_Layer5_1st_Iter + Output_Image7_Layer5_1st_Iter + Output_Image8_Layer5_1st_Iter    

        OutImage_1st_Layer5 = torch.tensor([float(value) for value in OutImages_1st_Layer5], dtype=torch.float32).reshape(8, 512, 13, 13)

        # Mean, Var
        s = time.time()
        Mean_1st_Layer5, Var_1st_Layer5 = Cal_mean_var.forward(OutImage_1st_Layer5)
        e = time.time()
        if DEBUG: print("Calculate Mean & Var : ",e-s)

        Beta_Layer5 = data.Beta_Dec[5]
        Gamma_Layer5 = data.Gamma_Dec[5]

        layer5_cache = BN(OutImage_1st_Layer5, Gamma_Layer5, Beta_Layer5)

        # Squeeze to remove the dimension but keeping the same data ordering
        Var_1st_Layer5 = Var_1st_Layer5.squeeze() * Gamma_Layer5

        s =time.time()
        Mean_1st_Layer5, Var_1st_Layer5 = Mean_Var_Dec2Bfloat(Mean_1st_Layer5, Var_1st_Layer5, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat : ",e-s)
        s = time.time()
        Weight_2nd_Layer5 = New_Weight_Hardware_ReOrdering_OtherLayer(512, 256, data.Weight_Bfloat[5], Mean_1st_Layer5, Var_1st_Layer5, data.Beta_Bfloat[5], Iteration="2")
        e = time.time()
        if DEBUG: print("Weight Reordering : ",e-s)

        '''
        data_read_mean_var = "result/layer5_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer5:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")    
        '''            

        # Write DDR
        s = time.time()
        Write_DDR(data_256_32(Weight_2nd_Layer5[0]), Wr_Address=0x8006AE00)
        Write_DDR(data_256_32(Weight_2nd_Layer5[1]), Wr_Address=0x9006AE00)
        e = time.time()
        if DEBUG: print("Write DDR & 256bit to 32bit : ",e-s)

        layer5_end = time.time()
        layer5_process = layer5_end - Layer5_start
        if DEBUG: print("Layer5 process time : ", layer5_process)
        
        resume()
        #if DEBUG: print(irq_val)

        '''
        d = Device("0000:08:00.0")
        bar = d.bar[0]

        data_read = open("result/layer5_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        d = Device("0000:08:00.0")
        bar = d.bar[2]

        data_read = open("result/layer5_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X6B18000-0X6A48000)/4) ): 
            Read_Data = bar.read(0X6A48000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer5_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X6B18000-0X6A48000)/4) ): 
            Read_Data = bar.read(0X16A48000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer5_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X7F34000-0X7E64000)/4) ): 
            Read_Data = bar.read(0X7E64000 + (i*4))
            data_read.write(str(Read_Data) + "\n")     
        '''

        #################################################
        #                Layer 6 Start                  #
        #################################################
        # check Layer6 IRQ
        check_irq_otherlayer()
        # self.app_instance .change_color(self.app_instance.L7_IRQ_canvas, self.app_instance.L7_IRQ, "green")
        # Layer 6
        Layer6_start = time.time()
        s = time.time()
        # Read DDR & Conver Format # 512MB
        Layer6_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x86B18000, End_Address=0x86B4C000)
        Layer6_1st_Iter_Image1_CH0_256 = (data_32_to_16(Layer6_1st_Iter_Image1_CH0))   
        #if DEBUG: print("ch0 image 1 : ", len(Layer6_1st_Iter_Image1_CH0))     

        Layer6_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x86B4C000, End_Address=0x86B80000)
        Layer6_1st_Iter_Image2_CH0_256 = (data_32_to_16(Layer6_1st_Iter_Image2_CH0))
        #if DEBUG: print("ch0 image 2 : ", len(Layer6_1st_Iter_Image2_CH0))
        
        Layer6_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x86B80000, End_Address=0x86BB4000)
        Layer6_1st_Iter_Image3_CH0_256 = (data_32_to_16(Layer6_1st_Iter_Image3_CH0))
        #if DEBUG: print("ch0 image 3 : ", len(Layer6_1st_Iter_Image3_CH0))

        Layer6_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x86BB4000, End_Address=0x86BE8000)
        Layer6_1st_Iter_Image4_CH0_256 = (data_32_to_16(Layer6_1st_Iter_Image4_CH0))
        #if DEBUG: print("ch0 image 4 : ", len(Layer6_1st_Iter_Image4_CH0))

        Layer6_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x86BE8000, End_Address=0x86C1C000)
        Layer6_1st_Iter_Image5_CH0_256 = (data_32_to_16(Layer6_1st_Iter_Image5_CH0))
        #if DEBUG: print("ch0 image 5 : ", len(Layer6_1st_Iter_Image5_CH0))

        Layer6_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x86C1C000, End_Address=0x86C50000)
        Layer6_1st_Iter_Image6_CH0_256 = (data_32_to_16(Layer6_1st_Iter_Image6_CH0))
        #if DEBUG: print("ch0 image 6 : ", len(Layer6_1st_Iter_Image6_CH0))

        Layer6_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x86C50000, End_Address=0x86C84000)
        Layer6_1st_Iter_Image7_CH0_256 = (data_32_to_16(Layer6_1st_Iter_Image7_CH0))
        #if DEBUG: print("ch0 image 7 : ", len(Layer6_1st_Iter_Image7_CH0))

        Layer6_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x86C84000, End_Address=0x86CB8000)
        Layer6_1st_Iter_Image8_CH0_256 = (data_32_to_16(Layer6_1st_Iter_Image8_CH0))
        #if DEBUG: print("ch0 image 8 : ", len(Layer6_1st_Iter_Image8_CH0))


        Layer6_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x96B18000, End_Address=0x96B4C000)
        Layer6_1st_Iter_Image1_CH1_256 = (data_32_to_16(Layer6_1st_Iter_Image1_CH1))
        #if DEBUG: print("ch1 image 1 : ", len(Layer6_1st_Iter_Image1_CH1))

        Layer6_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x96B4C000, End_Address=0x96B80000)
        Layer6_1st_Iter_Image2_CH1_256 = (data_32_to_16(Layer6_1st_Iter_Image2_CH1))
        #if DEBUG: print("ch1 image 2 : ", len(Layer6_1st_Iter_Image2_CH1))

        Layer6_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x96B80000, End_Address=0x96BB4000)
        Layer6_1st_Iter_Image3_CH1_256 = (data_32_to_16(Layer6_1st_Iter_Image3_CH1))
        #if DEBUG: print("ch1 image 3 : ", len(Layer6_1st_Iter_Image3_CH1))

        Layer6_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x96BB4000, End_Address=0x96BE8000)
        Layer6_1st_Iter_Image4_CH1_256 = (data_32_to_16(Layer6_1st_Iter_Image4_CH1))
        #if DEBUG: print("ch1 image 4 : ", len(Layer6_1st_Iter_Image4_CH1))

        Layer6_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x96BE8000, End_Address=0x96C1C000)
        Layer6_1st_Iter_Image5_CH1_256 = (data_32_to_16(Layer6_1st_Iter_Image5_CH1))
        #if DEBUG: print("ch1 image 5 : ", len(Layer6_1st_Iter_Image5_CH1))

        Layer6_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x96C1C000, End_Address=0x96C50000)
        Layer6_1st_Iter_Image6_CH1_256 = (data_32_to_16(Layer6_1st_Iter_Image6_CH1))
        #if DEBUG: print("ch1 image 6 : ", len(Layer6_1st_Iter_Image6_CH1))

        Layer6_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x96C50000, End_Address=0x96C84000)
        Layer6_1st_Iter_Image7_CH1_256 = (data_32_to_16(Layer6_1st_Iter_Image7_CH1))
        #if DEBUG: print("ch1 image 7 : ", len(Layer6_1st_Iter_Image7_CH1))

        Layer6_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x96C84000, End_Address=0x96CB8000)
        Layer6_1st_Iter_Image8_CH1_256 = (data_32_to_16(Layer6_1st_Iter_Image8_CH1))
        #if DEBUG: print("ch1 image 8 : ", len(Layer6_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR & 32bit to 16bit : ",e-s)


        '''
        test_out = '1st_iter_result/Layer6_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer6_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer6_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''
        
        s = time.time()
        Output_Image1_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image1_CH0_256, Layer6_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image2_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image2_CH0_256, Layer6_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image3_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image3_CH0_256, Layer6_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image4_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image4_CH0_256, Layer6_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image5_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image5_CH0_256, Layer6_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image6_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image6_CH0_256, Layer6_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image7_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image7_CH0_256, Layer6_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image8_Layer6_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer6_1st_Iter_Image8_CH0_256, Layer6_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)

        OutImages_1st_Layer6 = Output_Image1_Layer6_1st_Iter + Output_Image2_Layer6_1st_Iter + Output_Image3_Layer6_1st_Iter + Output_Image4_Layer6_1st_Iter + \
                            Output_Image5_Layer6_1st_Iter + Output_Image6_Layer6_1st_Iter + Output_Image7_Layer6_1st_Iter + Output_Image8_Layer6_1st_Iter    

        OutImage_1st_Layer6 = torch.tensor([float(value) for value in OutImages_1st_Layer6], dtype=torch.float32).reshape(8, 1024, 13, 13)

        # Mean, Var
        Mean_1st_Layer6, Var_1st_Layer6 = Cal_mean_var.forward(OutImage_1st_Layer6)
        
        Beta_Layer6 = data.Beta_Dec[6]
        Gamma_Layer6 = data.Gamma_Dec[6]

        layer6_cache = BN(OutImage_1st_Layer6, Gamma_Layer6, Beta_Layer6)

        # Squeeze to remove the dimension but keeping the same data ordering
        Var_1st_Layer6 = Var_1st_Layer6.squeeze() * Gamma_Layer6

        s = time.time()
        Mean_1st_Layer6, Var_1st_Layer6 = Mean_Var_Dec2Bfloat(Mean_1st_Layer6, Var_1st_Layer6, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat : ",e-s)
        s = time.time()
        Weight_2nd_Layer6 = New_Weight_Hardware_ReOrdering_OtherLayer(1024, 512, data.Weight_Bfloat[6], Mean_1st_Layer6, Var_1st_Layer6, data.Beta_Bfloat[6], Iteration="2")
        e = time.time()
        if DEBUG: print("Weight Reordering : ",e-s)

        '''
        data_read_mean_var = "result/layer6_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer6:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")        
        '''        

        # Write DDR
        s = time.time()
        Write_DDR(data_256_32(Weight_2nd_Layer6[0]), Wr_Address=0x801AAE00)
        Write_DDR(data_256_32(Weight_2nd_Layer6[1]), Wr_Address=0x901AAE00)
        e = time.time()
        if DEBUG: print("Write DDR & 256bit to 32bit : ",e-s)

        layer6_end = time.time()
        layer6_process = layer6_end - Layer6_start
        if DEBUG: print("Layer6 process time : ", layer6_process)

        resume()

        '''
        d = Device("0000:08:00.0")
        bar = d.bar[0]

        data_read = open("result/layer6_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        d = Device("0000:08:00.0")
        bar = d.bar[2]

        data_read = open("result/layer6_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X6E58000-0X6CB8000)/4) ): 
            Read_Data = bar.read(0X6CB8000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer6_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X6E58000-0X6CB8000)/4) ): 
            Read_Data = bar.read(0X16CB8000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer6_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X80D4000-0X7F34000)/4) ): 
            Read_Data = bar.read(0X7F34000 + (i*4))
            data_read.write(str(Read_Data) + "\n")       
        '''

        #################################################
        #                Layer 7 Start                  #
        #################################################
        # check Layer7 IRQ
        check_irq_otherlayer()
        # self.app_instance .change_color(self.app_instance.L8_IRQ_canvas, self.app_instance.L8_IRQ, "green")
        # Layer 7
        Layer7_start = time.time()
        s = time.time()
        # Read DDR & Conver Format # 512MB
        # if DEBUG: print("Read DDR")
        Layer7_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x86E58000, End_Address=0x86E8C000)
        Layer7_1st_Iter_Image1_CH0_256 = (data_32_to_16(Layer7_1st_Iter_Image1_CH0))   
        #if DEBUG: print("ch0 image 1 : ", len(Layer7_1st_Iter_Image1_CH0))     

        Layer7_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x86E8C000, End_Address=0x86EC0000)
        Layer7_1st_Iter_Image2_CH0_256 = (data_32_to_16(Layer7_1st_Iter_Image2_CH0))
        #if DEBUG: print("ch0 image 2 : ", len(Layer7_1st_Iter_Image2_CH0))
        
        Layer7_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x86EC0000, End_Address=0x86EF4000)
        Layer7_1st_Iter_Image3_CH0_256 = (data_32_to_16(Layer7_1st_Iter_Image3_CH0))
        #if DEBUG: print("ch0 image 3 : ", len(Layer7_1st_Iter_Image3_CH0))

        Layer7_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x86EF4000, End_Address=0x86F28000)
        Layer7_1st_Iter_Image4_CH0_256 = (data_32_to_16(Layer7_1st_Iter_Image4_CH0))
        #if DEBUG: print("ch0 image 4 : ", len(Layer7_1st_Iter_Image4_CH0))

        Layer7_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x86F28000, End_Address=0x86F5C000)
        Layer7_1st_Iter_Image5_CH0_256 = (data_32_to_16(Layer7_1st_Iter_Image5_CH0))
        #if DEBUG: print("ch0 image 5 : ", len(Layer7_1st_Iter_Image5_CH0))

        Layer7_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x86F5C000, End_Address=0x86F90000)
        Layer7_1st_Iter_Image6_CH0_256 = (data_32_to_16(Layer7_1st_Iter_Image6_CH0))
        #if DEBUG: print("ch0 image 6 : ", len(Layer7_1st_Iter_Image6_CH0))

        Layer7_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x86F90000, End_Address=0x86FC4000)
        Layer7_1st_Iter_Image7_CH0_256 = (data_32_to_16(Layer7_1st_Iter_Image7_CH0))
        #if DEBUG: print("ch0 image 7 : ", len(Layer7_1st_Iter_Image7_CH0))

        Layer7_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x86FC4000, End_Address=0x86FF8000)
        Layer7_1st_Iter_Image8_CH0_256 = (data_32_to_16(Layer7_1st_Iter_Image8_CH0))
        #if DEBUG: print("ch0 image 8 : ", len(Layer7_1st_Iter_Image8_CH0))


        Layer7_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x96E58000, End_Address=0x96E8C000)
        Layer7_1st_Iter_Image1_CH1_256 = (data_32_to_16(Layer7_1st_Iter_Image1_CH1))
        #if DEBUG: print("ch1 image 1 : ", len(Layer7_1st_Iter_Image1_CH1))

        Layer7_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x96E8C000, End_Address=0x96EC0000)
        Layer7_1st_Iter_Image2_CH1_256 = (data_32_to_16(Layer7_1st_Iter_Image2_CH1))
        #if DEBUG: print("ch1 image 2 : ", len(Layer7_1st_Iter_Image2_CH1))

        Layer7_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x96EC0000, End_Address=0x96EF4000)
        Layer7_1st_Iter_Image3_CH1_256 = (data_32_to_16(Layer7_1st_Iter_Image3_CH1))
        #if DEBUG: print("ch1 image 3 : ", len(Layer7_1st_Iter_Image3_CH1))

        Layer7_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x96EF4000, End_Address=0x96F28000)
        Layer7_1st_Iter_Image4_CH1_256 = (data_32_to_16(Layer7_1st_Iter_Image4_CH1))
        #if DEBUG: print("ch1 image 4 : ", len(Layer7_1st_Iter_Image4_CH1))

        Layer7_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x96F28000, End_Address=0x96F5C000)
        Layer7_1st_Iter_Image5_CH1_256 = (data_32_to_16(Layer7_1st_Iter_Image5_CH1))
        #if DEBUG: print("ch1 image 5 : ", len(Layer7_1st_Iter_Image5_CH1))

        Layer7_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x96F5C000, End_Address=0x96F90000)
        Layer7_1st_Iter_Image6_CH1_256 = (data_32_to_16(Layer7_1st_Iter_Image6_CH1))
        #if DEBUG: print("ch1 image 6 : ", len(Layer7_1st_Iter_Image6_CH1))

        Layer7_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x96F90000, End_Address=0x96FC4000)
        Layer7_1st_Iter_Image7_CH1_256 = (data_32_to_16(Layer7_1st_Iter_Image7_CH1))
        #if DEBUG: print("ch1 image 7 : ", len(Layer7_1st_Iter_Image7_CH1))

        Layer7_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x96FC4000, End_Address=0x96FF8000)
        Layer7_1st_Iter_Image8_CH1_256 = (data_32_to_16(Layer7_1st_Iter_Image8_CH1))
        #if DEBUG: print("ch1 image 8 : ", len(Layer7_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR Time : ",e-s)


        '''
        test_out = '1st_iter_result/Layer7_1st_Iter_Image1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = '1st_iter_result/Layer7_1st_Iter_Image8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Layer7_1st_Iter_Image8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''
        s = time.time()
        Output_Image1_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image1_CH0_256, Layer7_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image2_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image2_CH0_256, Layer7_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image3_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image3_CH0_256, Layer7_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image4_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image4_CH0_256, Layer7_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image5_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image5_CH0_256, Layer7_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image6_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image6_CH0_256, Layer7_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image7_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image7_CH0_256, Layer7_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Image8_Layer7_1st_Iter = Read_OutFmap_Bfloat2Dec(Layer7_1st_Iter_Image8_CH0_256, Layer7_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)

        OutImages_1st_Layer7 = Output_Image1_Layer7_1st_Iter + Output_Image2_Layer7_1st_Iter + Output_Image3_Layer7_1st_Iter + Output_Image4_Layer7_1st_Iter + \
                            Output_Image5_Layer7_1st_Iter + Output_Image6_Layer7_1st_Iter + Output_Image7_Layer7_1st_Iter + Output_Image8_Layer7_1st_Iter    

        OutImage_1st_Layer7 = torch.tensor([float(value) for value in OutImages_1st_Layer7], dtype=torch.float32).reshape(8, 1024, 13, 13)
        e = time.time()
        # if DEBUG: print("OutFmap_Bfloat2Dec Convert Time : ", e-s)

        # Mean, Var
        s = time.time()
        Mean_1st_Layer7, Var_1st_Layer7 = Cal_mean_var.forward(OutImage_1st_Layer7)
        e = time.time()
        if DEBUG: print("Calculate Mean & Var Time : ",e-s)

        Beta_Layer7 = data.Beta_Dec[7]
        Gamma_Layer7 = data.Gamma_Dec[7]

        layer7_cache = BN(OutImage_1st_Layer7, Gamma_Layer7, Beta_Layer7)

        # Squeeze to remove the dimension but keeping the same data ordering
        Var_1st_Layer7 = Var_1st_Layer7.squeeze() * Gamma_Layer7

        s = time.time()
        Mean_1st_Layer7, Var_1st_Layer7 = Mean_Var_Dec2Bfloat(Mean_1st_Layer7, Var_1st_Layer7, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat : ",e-s)
        s = time.time()
        Weight_2nd_Layer7 = New_Weight_Hardware_ReOrdering_OtherLayer(1024, 1024, data.Weight_Bfloat[7], Mean_1st_Layer7, Var_1st_Layer7, data.Beta_Bfloat[7], Iteration="2")
        e = time.time()
        if DEBUG: print("New_Weight_Hardware_ReOrdering_OtherLayer Time : ", e-s)

        '''
        data_read_mean_var = "result/layer7_mean_var.txt"
        with open(data_read_mean_var, mode="w") as output_file:
            for sublist in Weight_2nd_Layer7:
                cleaned_sublist = [clean_string(item) for item in sublist]
                output_file.write(" ".join(map(str, cleaned_sublist)) + "\n")        
        
        '''

        # Write DDR
        s = time.time()
        Write_DDR(data_256_32(Weight_2nd_Layer7[0]), Wr_Address=0x806AAE00)
        Write_DDR(data_256_32(Weight_2nd_Layer7[1]), Wr_Address=0x906AAE00)
        e = time.time()
        if DEBUG: print("256 to 32 & Write DDR Time : ",e-s)

        layer7_end = time.time()
        layer7_process = layer7_end - Layer7_start
        if DEBUG: print("Layer7 process time : ", layer7_process)
        
        resume()

        '''
        d = Device("0000:08:00.0")
        bar = d.bar[0]

        data_read = open("result/layer7_slave.txt", mode="w+")
        i=0
        for i in range(0,16): 
            Read_Data = bar.read(0X00 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        d = Device("0000:08:00.0")
        bar = d.bar[2]

        data_read = open("result/layer7_result_ch0.txt", mode="w+")
        i=0
        for i in range(0,int((0X7198000-0X6FF8000)/4) ): 
            Read_Data = bar.read(0X6FF8000 + (i*4))
            data_read.write(str(Read_Data) + "\n")      

        data_read = open("result/layer7_result_ch1.txt", mode="w+")
        i=0
        for i in range(0,int((0X7198000-0X6FF8000)/4) ): 
            Read_Data = bar.read(0X16FF8000 + (i*4))
            data_read.write(str(Read_Data) + "\n") 

        data_read = open("result/layer7_location_result.txt", mode="w+")
        i=0
        for i in range(0,int((0X8274000-0X80D4000)/4) ): 
            Read_Data = bar.read(0X80D4000 + (i*4))
            data_read.write(str(Read_Data) + "\n")     
        '''    
        
        end = time.time()
        process_time = (end-start)/60
        # if DEBUG: print(f'Whole Process: {process_time} mn')
        #################################################
        #                Layer 8 Start                  #
        #################################################
        check_irq_otherlayer()
        # self.app_instance .change_color(self.app_instance.L9_IRQ_canvas, self.app_instance.L9_IRQ, "green")
        layer8_start = time.time()
        s = time.time()
        # Post-Processing Pre-Defined Conditions
        #Post_Start_Signal = "1"

        # OutputImage from Hardware

        # Post Processing
        #if Post_Start_Signal == "1" or Post_Start_Signal == "1".zfill(4) or Post_Start_Signal == "1".zfill(16):  

        # Layer 8
        
        # Read DDR & Conver Format # 512MB
        
        Layer8_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x87198000, End_Address=0x8719E800)
        Layer8_1st_Iter_Image1_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image1_CH0))   
        #if DEBUG: print("ch0 image 1 : ", len(Layer8_1st_Iter_Image1_CH0))     

        Layer8_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x8719E800, End_Address=0x871A5000)
        Layer8_1st_Iter_Image2_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image2_CH0))
        #if DEBUG: print("ch0 image 2 : ", len(Layer8_1st_Iter_Image2_CH0))

        Layer8_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x871A5000, End_Address=0x871AB800)
        Layer8_1st_Iter_Image3_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image3_CH0))
        #if DEBUG: print("ch0 image 3 : ", len(Layer8_1st_Iter_Image3_CH0))

        Layer8_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x871AB800, End_Address=0x871B2000)
        Layer8_1st_Iter_Image4_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image4_CH0))
        #if DEBUG: print("ch0 image 4 : ", len(Layer8_1st_Iter_Image4_CH0))

        Layer8_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x871B2000, End_Address=0x871B8800)
        Layer8_1st_Iter_Image5_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image5_CH0))
        #if DEBUG: print("ch0 image 5 : ", len(Layer8_1st_Iter_Image5_CH0))

        Layer8_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x871B8800, End_Address=0x871BF000)
        Layer8_1st_Iter_Image6_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image6_CH0))
        #if DEBUG: print("ch0 image 6 : ", len(Layer8_1st_Iter_Image6_CH0))

        Layer8_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x871BF000, End_Address=0x871C5800)
        Layer8_1st_Iter_Image7_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image7_CH0))
        #if DEBUG: print("ch0 image 7 : ", len(Layer8_1st_Iter_Image7_CH0))

        Layer8_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x871C5800, End_Address=0x871CC000)
        Layer8_1st_Iter_Image8_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image8_CH0))
        #if DEBUG: print("ch0 image 8 : ", len(Layer8_1st_Iter_Image8_CH0))


        Layer8_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x97198000, End_Address=0x9719E800)
        Layer8_1st_Iter_Image1_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image1_CH1))   
        #if DEBUG: print("ch1 image 1 : ", len(Layer8_1st_Iter_Image1_CH1))     

        Layer8_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x9719E800, End_Address=0x971A5000)
        Layer8_1st_Iter_Image2_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image2_CH1))
        #if DEBUG: print("ch1 image 2 : ", len(Layer8_1st_Iter_Image2_CH1))

        Layer8_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x971A5000, End_Address=0x971AB800)
        Layer8_1st_Iter_Image3_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image3_CH1))
        #if DEBUG: print("ch1 image 3 : ", len(Layer8_1st_Iter_Image3_CH1))

        Layer8_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x971AB800, End_Address=0x971B2000)
        Layer8_1st_Iter_Image4_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image4_CH1))
        #if DEBUG: print("ch1 image 4 : ", len(Layer8_1st_Iter_Image4_CH1))

        Layer8_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x971B2000, End_Address=0x971B8800)
        Layer8_1st_Iter_Image5_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image5_CH1))
        #if DEBUG: print("ch1 image 5 : ", len(Layer8_1st_Iter_Image5_CH1))

        Layer8_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x971B8800, End_Address=0x971BF000)
        Layer8_1st_Iter_Image6_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image6_CH1))
        #if DEBUG: print("ch1 image 6 : ", len(Layer8_1st_Iter_Image6_CH1))

        Layer8_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x971BF000, End_Address=0x971C5800)
        Layer8_1st_Iter_Image7_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image7_CH1))
        #if DEBUG: print("ch1 image 7 : ", len(Layer8_1st_Iter_Image7_CH1))

        Layer8_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x971C5800, End_Address=0x971CC000)
        Layer8_1st_Iter_Image8_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image8_CH1))
        #if DEBUG: print("ch1 image 8 : ", len(Layer8_1st_Iter_Image8_CH1))
        e = time.time()
        if DEBUG: print("Read DDR & 32bit to 16bit : ",e-s)

        Output_Image1 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image1_CH0_256, Layer8_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image2 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image2_CH0_256, Layer8_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image3 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image3_CH0_256, Layer8_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image4 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image4_CH0_256, Layer8_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image5 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image5_CH0_256, Layer8_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image6 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image6_CH0_256, Layer8_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image7 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image7_CH0_256, Layer8_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image8 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image8_CH0_256, Layer8_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Layer8 = Output_Image1 + Output_Image2 + Output_Image3 + Output_Image4 + \
                        Output_Image5 + Output_Image6 + Output_Image7 + Output_Image8

        Float_OutputImage = [np.float32(x) for x in Output_Layer8]
        Float_OutputImage = Float_OutputImage[0:(8*125*(13**2))]
        Output_Layer8 = torch.tensor(Float_OutputImage, requires_grad=True).reshape(8,125, 13, 13)
        
        if DEBUG2 : Save_File(Output_Layer8, "result/Layer_8_Forward")   
        if DEBUG2 : Save_File(data.Bias_Dec[8],   "result/Layer_8_Forward_Bias_Before_Weight_Update")
        if DEBUG2 : Save_File(data.Weight_Dec[8], "result/Layer_8_Forward_weight_Before_Weight_Update")
        resume()
        
        return Output_Layer8

   
     
    def Forward_Inference(self, data):

        #################################################
        #                Layer 8 Start                  #
        #################################################
        # Read DDR & Conver Format # 512MB
        
        Layer8_1st_Iter_Image1_CH0 = Read_DDR(Rd_Address=0x87198000, End_Address=0x8719E800)
        Layer8_1st_Iter_Image1_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image1_CH0))   
        #if DEBUG: print("ch0 image 1 : ", len(Layer8_1st_Iter_Image1_CH0))     

        Layer8_1st_Iter_Image2_CH0 = Read_DDR(Rd_Address=0x8719E800, End_Address=0x871A5000)
        Layer8_1st_Iter_Image2_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image2_CH0))
        #if DEBUG: print("ch0 image 2 : ", len(Layer8_1st_Iter_Image2_CH0))

        Layer8_1st_Iter_Image3_CH0 = Read_DDR(Rd_Address=0x871A5000, End_Address=0x871AB800)
        Layer8_1st_Iter_Image3_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image3_CH0))
        #if DEBUG: print("ch0 image 3 : ", len(Layer8_1st_Iter_Image3_CH0))

        Layer8_1st_Iter_Image4_CH0 = Read_DDR(Rd_Address=0x871AB800, End_Address=0x871B2000)
        Layer8_1st_Iter_Image4_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image4_CH0))
        #if DEBUG: print("ch0 image 4 : ", len(Layer8_1st_Iter_Image4_CH0))

        Layer8_1st_Iter_Image5_CH0 = Read_DDR(Rd_Address=0x871B2000, End_Address=0x871B8800)
        Layer8_1st_Iter_Image5_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image5_CH0))
        #if DEBUG: print("ch0 image 5 : ", len(Layer8_1st_Iter_Image5_CH0))

        Layer8_1st_Iter_Image6_CH0 = Read_DDR(Rd_Address=0x871B8800, End_Address=0x871BF000)
        Layer8_1st_Iter_Image6_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image6_CH0))
        #if DEBUG: print("ch0 image 6 : ", len(Layer8_1st_Iter_Image6_CH0))

        Layer8_1st_Iter_Image7_CH0 = Read_DDR(Rd_Address=0x871BF000, End_Address=0x871C5800)
        Layer8_1st_Iter_Image7_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image7_CH0))
        #if DEBUG: print("ch0 image 7 : ", len(Layer8_1st_Iter_Image7_CH0))

        Layer8_1st_Iter_Image8_CH0 = Read_DDR(Rd_Address=0x871C5800, End_Address=0x871CC000)
        Layer8_1st_Iter_Image8_CH0_256 = (data_32_to_16(Layer8_1st_Iter_Image8_CH0))
        #if DEBUG: print("ch0 image 8 : ", len(Layer8_1st_Iter_Image8_CH0))


        Layer8_1st_Iter_Image1_CH1 = Read_DDR(Rd_Address=0x97198000, End_Address=0x9719E800)
        Layer8_1st_Iter_Image1_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image1_CH1))   
        #if DEBUG: print("ch1 image 1 : ", len(Layer8_1st_Iter_Image1_CH1))     

        Layer8_1st_Iter_Image2_CH1 = Read_DDR(Rd_Address=0x9719E800, End_Address=0x971A5000)
        Layer8_1st_Iter_Image2_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image2_CH1))
        #if DEBUG: print("ch1 image 2 : ", len(Layer8_1st_Iter_Image2_CH1))

        Layer8_1st_Iter_Image3_CH1 = Read_DDR(Rd_Address=0x971A5000, End_Address=0x971AB800)
        Layer8_1st_Iter_Image3_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image3_CH1))
        #if DEBUG: print("ch1 image 3 : ", len(Layer8_1st_Iter_Image3_CH1))

        Layer8_1st_Iter_Image4_CH1 = Read_DDR(Rd_Address=0x971AB800, End_Address=0x971B2000)
        Layer8_1st_Iter_Image4_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image4_CH1))
        #if DEBUG: print("ch1 image 4 : ", len(Layer8_1st_Iter_Image4_CH1))

        Layer8_1st_Iter_Image5_CH1 = Read_DDR(Rd_Address=0x971B2000, End_Address=0x971B8800)
        Layer8_1st_Iter_Image5_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image5_CH1))
        #if DEBUG: print("ch1 image 5 : ", len(Layer8_1st_Iter_Image5_CH1))

        Layer8_1st_Iter_Image6_CH1 = Read_DDR(Rd_Address=0x971B8800, End_Address=0x971BF000)
        Layer8_1st_Iter_Image6_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image6_CH1))
        #if DEBUG: print("ch1 image 6 : ", len(Layer8_1st_Iter_Image6_CH1))

        Layer8_1st_Iter_Image7_CH1 = Read_DDR(Rd_Address=0x971BF000, End_Address=0x971C5800)
        Layer8_1st_Iter_Image7_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image7_CH1))
        #if DEBUG: print("ch1 image 7 : ", len(Layer8_1st_Iter_Image7_CH1))

        Layer8_1st_Iter_Image8_CH1 = Read_DDR(Rd_Address=0x971C5800, End_Address=0x971CC000)
        Layer8_1st_Iter_Image8_CH1_256 = (data_32_to_16(Layer8_1st_Iter_Image8_CH1))
        #if DEBUG: print("ch1 image 8 : ", len(Layer8_1st_Iter_Image8_CH1))
        e = time.time()

        Output_Image1 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image1_CH0_256, Layer8_1st_Iter_Image1_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image2 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image2_CH0_256, Layer8_1st_Iter_Image2_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image3 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image3_CH0_256, Layer8_1st_Iter_Image3_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image4 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image4_CH0_256, Layer8_1st_Iter_Image4_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image5 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image5_CH0_256, Layer8_1st_Iter_Image5_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image6 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image6_CH0_256, Layer8_1st_Iter_Image6_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image7 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image7_CH0_256, Layer8_1st_Iter_Image7_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Image8 = OutFmap_Layer8_BFPtoDec(Layer8_1st_Iter_Image8_CH0_256, Layer8_1st_Iter_Image8_CH1_256, Exponent_Bits, Mantissa_Bits)
        Output_Layer8 = Output_Image1 + Output_Image2 + Output_Image3 + Output_Image4 + \
                        Output_Image5 + Output_Image6 + Output_Image7 + Output_Image8

        Float_OutputImage = [np.float32(x) for x in Output_Layer8]
        Float_OutputImage = Float_OutputImage[0:(8*125*(13**2))]
        Output_Layer8 = torch.tensor(Float_OutputImage, requires_grad=True).reshape(8,125, 13, 13)
        
        if DEBUG2 : Save_File(Output_Layer8, "result/Layer_8_Forward")   
        
        return Output_Layer8

    def Pre_Processing_Backward(self, data, Loss_Gradient):     
        
        # Weight_Backward_Layer8 for Soft2Hardware
        # if epoch == 0:
        layer8_start = time.time()
        s = time.time()
        Weight_Backward_Layer8 = Weight_Hardware_Backward_ReOrdering_Layer8(128, 1024, data.Weight_Bfloat[8]+["0000"]*3072, ["0000"]*128, ["0000"]*128)
        e = time.time()
        if DEBUG: print("Weight Reordering : ",e-s)
        
        # Break 256To32 and Flip the Data: 
        s = time.time()
        Weight_Backward_CH0 = data_256_32(Weight_Backward_Layer8[0])
        Weight_Backward_CH1 = data_256_32(Weight_Backward_Layer8[1])
        e = time.time()
        if DEBUG: print("256bit to 32bit : ",e-s)

        # Write Weight For Backward into DDR
        s = time.time()
        Write_DDR(Weight_Backward_CH0,Wr_Address=0x81200000)
        Write_DDR(Weight_Backward_CH1,Wr_Address=0x91200000)
        e = time.time()
        if DEBUG: print("Write DDR : ",e-s)
        
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
        s = time.time()
        Loss_Grad1_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient1_layer8, Exponent_Bits, Mantissa_Bits)
        Loss_Grad1_layer8 = Fmap_Ordering(Channel=128, Data_List=Loss_Grad1_layer8)
        # Loss_Grad2:
        Loss_Grad2_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient2_layer8, Exponent_Bits, Mantissa_Bits)
        Loss_Grad2_layer8 = Fmap_Ordering(Channel=128, Data_List=Loss_Grad2_layer8) 
        # Loss_Grad3:  
        Loss_Grad3_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient3_layer8, Exponent_Bits, Mantissa_Bits)
        Loss_Grad3_layer8 = Fmap_Ordering(Channel=128, Data_List=Loss_Grad3_layer8) 
        # Loss_Grad4:  
        Loss_Grad4_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient4_layer8, Exponent_Bits, Mantissa_Bits)
        Loss_Grad4_layer8 = Fmap_Ordering(Channel=128, Data_List=Loss_Grad4_layer8) 
        # Loss_Grad5:
        Loss_Grad5_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient5_layer8, Exponent_Bits, Mantissa_Bits)
        Loss_Grad5_layer8 = Fmap_Ordering(Channel=128, Data_List=Loss_Grad5_layer8) 
        # Loss_Grad6:
        Loss_Grad6_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient6_layer8, Exponent_Bits, Mantissa_Bits)
        Loss_Grad6_layer8 = Fmap_Ordering(Channel=128, Data_List=Loss_Grad6_layer8) 
        # Loss_Grad7:  
        Loss_Grad7_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient7_layer8, Exponent_Bits, Mantissa_Bits)
        Loss_Grad7_layer8 = Fmap_Ordering(Channel=128, Data_List=Loss_Grad7_layer8) 
        # Loss_Grad8:  
        Loss_Grad8_layer8 = Loss_Gradient_Dec2Bfloat(Loss_Gradient8_layer8, Exponent_Bits, Mantissa_Bits)
        Loss_Grad8_layer8 = Fmap_Ordering(Channel=128, Data_List=Loss_Grad8_layer8)
        e = time.time()
        if DEBUG: print("Loss Reordering : ",e-s)
        
        # Separate the DDR Channel: 
        Loss_Grad_layer8_CH0 =  Loss_Grad1_layer8[0] + Loss_Grad2_layer8[0] + Loss_Grad3_layer8[0] + Loss_Grad4_layer8[0] + \
                                Loss_Grad5_layer8[0] + Loss_Grad6_layer8[0] + Loss_Grad7_layer8[0] + Loss_Grad8_layer8[0]
        
        Loss_Grad_layer8_CH1 =  Loss_Grad1_layer8[1] + Loss_Grad2_layer8[1] + Loss_Grad3_layer8[1] + Loss_Grad4_layer8[1] + \
                                Loss_Grad5_layer8[1] + Loss_Grad6_layer8[1] + Loss_Grad7_layer8[1] + Loss_Grad8_layer8[1]
        

        # Write Loss Gradient to DDR:
        s = time.time()
        Write_DDR(data_256_32(Loss_Grad_layer8_CH0), Wr_Address=0x882A8000)
        Write_DDR(data_256_32(Loss_Grad_layer8_CH1), Wr_Address=0x982A8000)
        e = time.time()
        if DEBUG: print("Write DDR & 256bit to 32bit : ",e-s)

        layer8_end = time.time()
        if DEBUG: print("layer8 process : ",layer8_end-layer8_start)
        resume()
        #if DEBUG: print(irq_val)

        '''
        output_file1 = "result/Loss_Grad_layer8_CH0.txt"
        with open(output_file1, mode="w") as output_file_1:
            for item in Loss_Grad_layer8_CH0:
                output_file_1.write(str(item) + "\n")
        output_file2 = "result/Loss_Grad_layer8_CH1.txt"
        with open(output_file2, mode="w") as output_file_2:
            for item in Loss_Grad_layer8_CH1:
                output_file_2.write(str(item) + "\n")        
        output_file_1.close()
        output_file_2.close()
        '''    

        # Bias Gradient Calculation
        # Bias_Grad = torch.sum(Loss_Gradient, dim=(0, 2, 3))   
        # return Bias_Grad
        # return Loss
        

    def Backward(self, data, Loss_Gradient):
        
        Bias_Grad = torch.sum(Loss_Gradient, dim=(0, 2, 3)) 
        
        if DEBUG2 : Save_File(Bias_Grad, "result/Layer_8_Backward_Bias_Gradient")   
         
        Blayer8_start = time.time()
        #################################################
        #             Backward Layer 8 Start            #
        #################################################
        global Weight_Gradient_Layer0, Weight_Gradient_Layer1, Weight_Gradient_Layer2, Weight_Gradient_Layer3, Weight_Gradient_Layer4,\
            Weight_Gradient_Layer5, Weight_Gradient_Layer6, Weight_Gradient_Layer7, Weight_Gradient_Layer8
        global dL_dgamma_7, dL_dbeta_7, dL_dgamma_6, dL_dbeta_6, dL_dgamma_5, dL_dbeta_5, dL_dgamma_4, dL_dbeta_4, dL_dgamma_3, dL_dbeta_3, dL_dgamma_2, dL_dbeta_2, \
            dL_dgamma_1, dL_dbeta_1, dL_dgamma_0, dL_dbeta_0
        
        global Gamma_Gradient, Beta_Gradient, Weight_Gradient
                    
        # check Layer8 IRQ
        check_irq_otherlayer()       
        s = time.time()
        # self.app_instance .change_color(self.app_instance.L9_IRQ_canvas, self.app_instance.L9_IRQ, "green")
        # Read Gradient of Output After ReLU Backward: 
        Output_Grad1_Layer8_CH0 = Read_DDR(Rd_Address=0x86E58000,  End_Address=0x86E8C000)
        Output_Grad1_Layer8_CH0_16 = data_32_to_16(Output_Grad1_Layer8_CH0)
        #if DEBUG: print("Read Output_Grad1_Layer8_CH0")

        Output_Grad1_Layer8_CH1 = Read_DDR(Rd_Address=0x96E58000,  End_Address=0x96E8C000)
        Output_Grad1_Layer8_CH1_16 = data_32_to_16(Output_Grad1_Layer8_CH1)
        #if DEBUG: print("Read Output_Grad1_Layer8_CH1")
        
        Output_Grad2_Layer8_CH0 = Read_DDR(Rd_Address=0x86E8C000,  End_Address=0x86EC0000)
        Output_Grad2_Layer8_CH0_16 = data_32_to_16(Output_Grad2_Layer8_CH0)
        #if DEBUG: print("Read Output_Grad2_Layer8_CH0")

        Output_Grad2_Layer8_CH1 = Read_DDR(Rd_Address=0x96E8C000,  End_Address=0x96EC0000)
        Output_Grad2_Layer8_CH1_16 = data_32_to_16(Output_Grad2_Layer8_CH1)
        #if DEBUG: print("Read Output_Grad2_Layer8_CH1")

        Output_Grad3_Layer8_CH0 = Read_DDR(Rd_Address=0x86EC0000,  End_Address=0x86EF4000)
        Output_Grad3_Layer8_CH0_16 = data_32_to_16(Output_Grad3_Layer8_CH0)
        #if DEBUG: print("Read Output_Grad3_Layer8_CH0")

        Output_Grad3_Layer8_CH1 = Read_DDR(Rd_Address=0x96EC0000,  End_Address=0x96EF4000)
        Output_Grad3_Layer8_CH1_16 = data_32_to_16(Output_Grad3_Layer8_CH1)
        #if DEBUG: print("Read Output_Grad3_Layer8_CH1")

        Output_Grad4_Layer8_CH0 = Read_DDR(Rd_Address=0x86EF4000,  End_Address=0x86F28000)
        Output_Grad4_Layer8_CH0_16 = data_32_to_16(Output_Grad4_Layer8_CH0)
        #if DEBUG: print("Read Output_Grad4_Layer8_CH0")

        Output_Grad4_Layer8_CH1 = Read_DDR(Rd_Address=0x96EF4000,  End_Address=0x96F28000)
        Output_Grad4_Layer8_CH1_16 = data_32_to_16(Output_Grad4_Layer8_CH1)
        #if DEBUG: print("Read Output_Grad4_Layer8_CH1")

        Output_Grad5_Layer8_CH0 = Read_DDR(Rd_Address=0x86F28000,  End_Address=0x86F5C000)
        Output_Grad5_Layer8_CH0_16 = data_32_to_16(Output_Grad5_Layer8_CH0)
        #if DEBUG: print("Read Output_Grad5_Layer8_CH0")

        Output_Grad5_Layer8_CH1 = Read_DDR(Rd_Address=0x96F28000,  End_Address=0x96F5C000)
        Output_Grad5_Layer8_CH1_16 = data_32_to_16(Output_Grad5_Layer8_CH1)
        #if DEBUG: print("Read Output_Grad5_Layer8_CH1")

        Output_Grad6_Layer8_CH0 = Read_DDR(Rd_Address=0x86F5C000,  End_Address=0x86F90000)
        Output_Grad6_Layer8_CH0_16 = data_32_to_16(Output_Grad6_Layer8_CH0)
        #if DEBUG: print("Read Output_Grad6_Layer8_CH0")

        Output_Grad6_Layer8_CH1 = Read_DDR(Rd_Address=0x96F5C000,  End_Address=0x96F90000)
        Output_Grad6_Layer8_CH1_16 = data_32_to_16(Output_Grad6_Layer8_CH1)
        #if DEBUG: print("Read Output_Grad6_Layer8_CH1")

        Output_Grad7_Layer8_CH0 = Read_DDR(Rd_Address=0x86F90000,  End_Address=0x86FC4000)
        Output_Grad7_Layer8_CH0_16 = data_32_to_16(Output_Grad7_Layer8_CH0)
        #if DEBUG: print("Read Output_Grad7_Layer8_CH0")

        Output_Grad7_Layer8_CH1 = Read_DDR(Rd_Address=0x96F90000,  End_Address=0x96FC4000)
        Output_Grad7_Layer8_CH1_16 = data_32_to_16(Output_Grad7_Layer8_CH1)
        #if DEBUG: print("Read Output_Grad7_Layer8_CH1")

        Output_Grad8_Layer8_CH0 = Read_DDR(Rd_Address=0x86FC4000,  End_Address=0x86FF8000)
        Output_Grad8_Layer8_CH0_16 = data_32_to_16(Output_Grad8_Layer8_CH0)
        #if DEBUG: print("Read Output_Grad8_Layer8_CH0")

        Output_Grad8_Layer8_CH1 = Read_DDR(Rd_Address=0x96FC4000,  End_Address=0x96FF8000)
        Output_Grad8_Layer8_CH1_16 = data_32_to_16(Output_Grad8_Layer8_CH1)    
        #if DEBUG: print("Read Output_Grad8_Layer8_CH1")
        e = time.time()
        if DEBUG: print("Read OutG DDR & 32bit to 16bit : ",e-s)

        s = time.time()
        Output_Grad1_Layer8 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer8_CH0_16, Output_Grad1_Layer8_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad2_Layer8 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer8_CH0_16, Output_Grad2_Layer8_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad3_Layer8 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer8_CH0_16, Output_Grad3_Layer8_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad4_Layer8 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer8_CH0_16, Output_Grad4_Layer8_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad5_Layer8 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer8_CH0_16, Output_Grad5_Layer8_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad6_Layer8 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer8_CH0_16, Output_Grad6_Layer8_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad7_Layer8 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer8_CH0_16, Output_Grad7_Layer8_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad8_Layer8 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer8_CH0_16, Output_Grad8_Layer8_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)

        Output_Grads_Layer8 = Output_Grad1_Layer8 + Output_Grad2_Layer8 + Output_Grad3_Layer8 + Output_Grad4_Layer8 + \
                                Output_Grad5_Layer8 + Output_Grad6_Layer8 + Output_Grad7_Layer8 + Output_Grad8_Layer8    
        Output_Grad_Layer8 = torch.tensor([float(value) for value in Output_Grads_Layer8], dtype=torch.float32).reshape(8, 1024, 13, 13)
        
        if DEBUG2 : Save_File(Output_Grad_Layer8, "result/Layer_8_Backward_Input_Gradient")

        # BReLu Marking for Layer7
        s = time.time()
        ReLu_Marking1_Layer7_CH0 = Read_DDR(Rd_Address=0x880D4000,  End_Address=0x88108000)
        ReLu_Marking1_Layer7_CH0_256 = data_32_to_16(ReLu_Marking1_Layer7_CH0)

        ReLu_Marking2_Layer7_CH0 = Read_DDR(Rd_Address=0x88108000,  End_Address=0x8813C000)
        ReLu_Marking2_Layer7_CH0_256 = data_32_to_16(ReLu_Marking2_Layer7_CH0)

        ReLu_Marking3_Layer7_CH0 = Read_DDR(Rd_Address=0x8813C000,  End_Address=0x88170000)
        ReLu_Marking3_Layer7_CH0_256 = data_32_to_16(ReLu_Marking3_Layer7_CH0)

        ReLu_Marking4_Layer7_CH0 = Read_DDR(Rd_Address=0x88170000,  End_Address=0x881A4000)
        ReLu_Marking4_Layer7_CH0_256 = data_32_to_16(ReLu_Marking4_Layer7_CH0)

        ReLu_Marking5_Layer7_CH0 = Read_DDR(Rd_Address=0x881A4000,  End_Address=0x881D8000)
        ReLu_Marking5_Layer7_CH0_256 = data_32_to_16(ReLu_Marking5_Layer7_CH0)

        ReLu_Marking6_Layer7_CH0 = Read_DDR(Rd_Address=0x881D8000,  End_Address=0x8820C000)
        ReLu_Marking6_Layer7_CH0_256 = data_32_to_16(ReLu_Marking6_Layer7_CH0)

        ReLu_Marking7_Layer7_CH0 = Read_DDR(Rd_Address=0x8820C000,  End_Address=0x88240000)
        ReLu_Marking7_Layer7_CH0_256 = data_32_to_16(ReLu_Marking7_Layer7_CH0)

        ReLu_Marking8_Layer7_CH0 = Read_DDR(Rd_Address=0x88240000,  End_Address=0x88274000)
        ReLu_Marking8_Layer7_CH0_256 = data_32_to_16(ReLu_Marking8_Layer7_CH0)

        ReLu_Marking1_Layer7_CH1 = Read_DDR(Rd_Address=0x980D4000,  End_Address=0x98108000)
        ReLu_Marking1_Layer7_CH1_256 = data_32_to_16(ReLu_Marking1_Layer7_CH1)

        ReLu_Marking2_Layer7_CH1 = Read_DDR(Rd_Address=0x98108000,  End_Address=0x9813C000)
        ReLu_Marking2_Layer7_CH1_256 = data_32_to_16(ReLu_Marking2_Layer7_CH1)

        ReLu_Marking3_Layer7_CH1 = Read_DDR(Rd_Address=0x9813C000,  End_Address=0x98170000)
        ReLu_Marking3_Layer7_CH1_256 = data_32_to_16(ReLu_Marking3_Layer7_CH1)

        ReLu_Marking4_Layer7_CH1 = Read_DDR(Rd_Address=0x98170000,  End_Address=0x981A4000)
        ReLu_Marking4_Layer7_CH1_256 = data_32_to_16(ReLu_Marking4_Layer7_CH1)

        ReLu_Marking5_Layer7_CH1 = Read_DDR(Rd_Address=0x981A4000,  End_Address=0x981D8000)
        ReLu_Marking5_Layer7_CH1_256 = data_32_to_16(ReLu_Marking5_Layer7_CH1)

        ReLu_Marking6_Layer7_CH1 = Read_DDR(Rd_Address=0x981D8000,  End_Address=0x9820C000)
        ReLu_Marking6_Layer7_CH1_256 = data_32_to_16(ReLu_Marking6_Layer7_CH1)

        ReLu_Marking7_Layer7_CH1 = Read_DDR(Rd_Address=0x9820C000,  End_Address=0x98240000)
        ReLu_Marking7_Layer7_CH1_256 = data_32_to_16(ReLu_Marking7_Layer7_CH1)

        ReLu_Marking8_Layer7_CH1 = Read_DDR(Rd_Address=0x98240000,  End_Address=0x98274000)
        ReLu_Marking8_Layer7_CH1_256 = data_32_to_16(ReLu_Marking8_Layer7_CH1)
        e = time.time()
        if DEBUG: print("Read ReLu DDR & 32bit to 16bit : ",e-s)

        # ReLu Reordering
        s = time.time()
        ReLu_Marking1_Layer7 = Read_ReLu_Marking(ReLu_Marking1_Layer7_CH0_256, ReLu_Marking1_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        ReLu_Marking2_Layer7 = Read_ReLu_Marking(ReLu_Marking2_Layer7_CH0_256, ReLu_Marking2_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        ReLu_Marking3_Layer7 = Read_ReLu_Marking(ReLu_Marking3_Layer7_CH0_256, ReLu_Marking3_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        ReLu_Marking4_Layer7 = Read_ReLu_Marking(ReLu_Marking4_Layer7_CH0_256, ReLu_Marking4_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        ReLu_Marking5_Layer7 = Read_ReLu_Marking(ReLu_Marking5_Layer7_CH0_256, ReLu_Marking5_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        ReLu_Marking6_Layer7 = Read_ReLu_Marking(ReLu_Marking6_Layer7_CH0_256, ReLu_Marking6_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        ReLu_Marking7_Layer7 = Read_ReLu_Marking(ReLu_Marking7_Layer7_CH0_256, ReLu_Marking7_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        ReLu_Marking8_Layer7 = Read_ReLu_Marking(ReLu_Marking8_Layer7_CH0_256, ReLu_Marking8_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        e = time.time()
        if DEBUG: print("ReLU ordering : ",e-s)

        ReLu_Marking_Layer7 = ReLu_Marking1_Layer7 + ReLu_Marking2_Layer7 + ReLu_Marking3_Layer7 + ReLu_Marking4_Layer7 + ReLu_Marking5_Layer7 + \
                                ReLu_Marking6_Layer7 + ReLu_Marking7_Layer7 + ReLu_Marking8_Layer7
        
        ReLu_Marking_Layer7 = torch.tensor([float(value) for value in ReLu_Marking_Layer7], dtype=torch.float32).reshape(8, 1024, 13, 13)


        # BReLu Calculate
        s = time.time()
        # Output_Grad_layer8_input = torch.tensor(Output_Grad_Layer8, dtype=torch.float32).reshape(8,1024,13,13)
        # Layer7_Location = torch.tensor(ReLu_Marking_Layer7, dtype=torch.float32).reshape(8,1024,13,13)
        relu_mask, location_mask = split_location(ReLu_Marking_Layer7)
        grad_relu_output = backward_active(Output_Grad_Layer8, relu_mask)
        #grad_maxpool_output = backward_MaxPool_Location(grad_relu_output, location_mask)
        dL_dgamma_7, dL_dbeta_7, avg_pc_7, backward_const_7 = backward_LightNorm(grad_relu_output, layer7_cache)
        e = time.time()
        if DEBUG: print("Software Calculate : ",e-s)
        # avg_pc_7 = avg_pc_7.squeeze()
        # backward_const_7 = backward_const_7.squeeze()
        s = time.time()
        avg_pc_7, backward_const_7 = Mean_Var_Dec2Bfloat(avg_pc_7, backward_const_7, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat : ",e-s)
        
        
        if DEBUG2 : Save_File(dL_dgamma_7, "result/Layer_7_Backward_Gamma_Gradient")
        if DEBUG2 : Save_File(dL_dbeta_7,  "result/Layer_7_Backward_Beta_Gradient")

        # Weight Gradient
        s = time.time()
        Weight_Gradient1_Layer8_CH0 = Read_DDR(Rd_Address=0x882DC000,  End_Address=0x882FC000)
        Weight_Gradient1_Layer8_CH0_256 = data_32_to_16(Weight_Gradient1_Layer8_CH0)
        #if DEBUG: print("Weight_Gradient1_Layer8_CH0 : ", len(Weight_Gradient1_Layer8_CH0))   

        Weight_Gradient2_Layer8_CH0 = Read_DDR(Rd_Address=0x882FC000,  End_Address=0x8831C000)
        Weight_Gradient2_Layer8_CH0_256 = data_32_to_16(Weight_Gradient2_Layer8_CH0)
        #if DEBUG: print("Weight_Gradient2_Layer8_CH0 : ", len(Weight_Gradient2_Layer8_CH0))    

        Weight_Gradient3_Layer8_CH0 = Read_DDR(Rd_Address=0x8831C000,  End_Address=0x8833C000)
        Weight_Gradient3_Layer8_CH0_256 = data_32_to_16(Weight_Gradient3_Layer8_CH0)
        #if DEBUG: print("Weight_Gradient3_Layer8_CH0 : ", len(Weight_Gradient3_Layer8_CH0)) 

        Weight_Gradient4_Layer8_CH0 = Read_DDR(Rd_Address=0x8833C000,  End_Address=0x8835C000)
        Weight_Gradient4_Layer8_CH0_256 = data_32_to_16(Weight_Gradient4_Layer8_CH0)
        #if DEBUG: print("Weight_Gradient4_Layer8_CH0 : ", len(Weight_Gradient4_Layer8_CH0)) 

        Weight_Gradient5_Layer8_CH0 = Read_DDR(Rd_Address=0x8835C000,  End_Address=0X8837C000)
        Weight_Gradient5_Layer8_CH0_256 = data_32_to_16(Weight_Gradient5_Layer8_CH0)
        #if DEBUG: print("Weight_Gradient5_Layer8_CH0 : ", len(Weight_Gradient5_Layer8_CH0)) 

        Weight_Gradient6_Layer8_CH0 = Read_DDR(Rd_Address=0X8837C000,  End_Address=0x8839C000)
        Weight_Gradient6_Layer8_CH0_256 = data_32_to_16(Weight_Gradient6_Layer8_CH0)
        #if DEBUG: print("Weight_Gradient6_Layer8_CH0 : ", len(Weight_Gradient6_Layer8_CH0)) 

        Weight_Gradient7_Layer8_CH0 = Read_DDR(Rd_Address=0x8839C000,  End_Address=0x883BC000)
        Weight_Gradient7_Layer8_CH0_256 = data_32_to_16(Weight_Gradient7_Layer8_CH0)
        #if DEBUG: print("Weight_Gradient7_Layer8_CH0 : ", len(Weight_Gradient7_Layer8_CH0)) 

        Weight_Gradient8_Layer8_CH0 = Read_DDR(Rd_Address=0x883BC000,  End_Address=0x883DC000)
        Weight_Gradient8_Layer8_CH0_256 = data_32_to_16(Weight_Gradient8_Layer8_CH0)
        #if DEBUG: print("Weight_Gradient8_Layer8_CH0 : ", len(Weight_Gradient8_Layer8_CH0)) 


        Weight_Gradient1_Layer8_CH1 = Read_DDR(Rd_Address=0x982DC000,  End_Address=0x982FC000)
        Weight_Gradient1_Layer8_CH1_256 = data_32_to_16(Weight_Gradient1_Layer8_CH1)
        #if DEBUG: print("Weight_Gradient1_Layer8_CH1 : ", len(Weight_Gradient1_Layer8_CH1))   

        Weight_Gradient2_Layer8_CH1 = Read_DDR(Rd_Address=0x982FC000,  End_Address=0x9831C000)
        Weight_Gradient2_Layer8_CH1_256 = data_32_to_16(Weight_Gradient2_Layer8_CH1)
        #if DEBUG: print("Weight_Gradient2_Layer8_CH1 : ", len(Weight_Gradient2_Layer8_CH1))    

        Weight_Gradient3_Layer8_CH1 = Read_DDR(Rd_Address=0x9831C000,  End_Address=0x9833C000)
        Weight_Gradient3_Layer8_CH1_256 = data_32_to_16(Weight_Gradient3_Layer8_CH1)
        #if DEBUG: print("Weight_Gradient3_Layer8_CH1 : ", len(Weight_Gradient3_Layer8_CH1)) 

        Weight_Gradient4_Layer8_CH1 = Read_DDR(Rd_Address=0x9833C000,  End_Address=0x9835C000)
        Weight_Gradient4_Layer8_CH1_256 = data_32_to_16(Weight_Gradient4_Layer8_CH1)
        #if DEBUG: print("Weight_Gradient4_Layer8_CH1 : ", len(Weight_Gradient4_Layer8_CH1)) 

        Weight_Gradient5_Layer8_CH1 = Read_DDR(Rd_Address=0x9835C000,  End_Address=0x9837C000)
        Weight_Gradient5_Layer8_CH1_256 = data_32_to_16(Weight_Gradient5_Layer8_CH1)
        #if DEBUG: print("Weight_Gradient5_Layer8_CH1 : ", len(Weight_Gradient5_Layer8_CH1)) 

        Weight_Gradient6_Layer8_CH1 = Read_DDR(Rd_Address=0x9837C000,  End_Address=0x9839C000)
        Weight_Gradient6_Layer8_CH1_256 = data_32_to_16(Weight_Gradient6_Layer8_CH1)
        #if DEBUG: print("Weight_Gradient6_Layer8_CH1 : ", len(Weight_Gradient6_Layer8_CH1)) 

        Weight_Gradient7_Layer8_CH1 = Read_DDR(Rd_Address=0x9839C000,  End_Address=0x983BC000)
        Weight_Gradient7_Layer8_CH1_256 = data_32_to_16(Weight_Gradient7_Layer8_CH1)
        #if DEBUG: print("Weight_Gradient7_Layer8_CH1 : ", len(Weight_Gradient7_Layer8_CH1)) 

        Weight_Gradient8_Layer8_CH1 = Read_DDR(Rd_Address=0x983BC000,  End_Address=0x983DC000)
        Weight_Gradient8_Layer8_CH1_256 = data_32_to_16(Weight_Gradient8_Layer8_CH1)
        #if DEBUG: print("Weight_Gradient8_Layer8_CH1 : ", len(Weight_Gradient8_Layer8_CH1)) 
        e = time.time()
        if DEBUG: print("Read WG DDR & 32bit to 16bit ",e-s)

        '''
        test_out = 'Weight_Result/Weight_Gradient1_Layer8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient1_Layer8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer8_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer8_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer8_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer8_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''

        # Weight_Gradient_Layer8_CH0 = [Weight_Gradient1_Layer8_CH0_256, Weight_Gradient2_Layer8_CH0_256, Weight_Gradient3_Layer8_CH0_256, Weight_Gradient4_Layer8_CH0_256, 
        #                         Weight_Gradient5_Layer8_CH0_256, Weight_Gradient6_Layer8_CH0_256, Weight_Gradient7_Layer8_CH0_256, Weight_Gradient8_Layer8_CH0_256]
        # Weight_Gradient_Layer8_CH1 = [Weight_Gradient1_Layer8_CH1_256, Weight_Gradient2_Layer8_CH1_256, Weight_Gradient3_Layer8_CH1_256, Weight_Gradient4_Layer8_CH1_256, 
        #                         Weight_Gradient5_Layer8_CH1_256, Weight_Gradient6_Layer8_CH1_256, Weight_Gradient7_Layer8_CH1_256, Weight_Gradient8_Layer8_CH1_256]
        # s = time.time()
        # Weight_Gradient_Layer8 = Read_WeightGradient_Bfloat2Dec_whole_image(Weight_Gradient_Layer8_CH0, Weight_Gradient_Layer8_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
        # e = time.time()
        # if DEBUG: print("Read_WeightGradient_Bfloat2Dec_whole_image Time : ", e-s)
        s = time.time()
        Weight_Gradient1_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer8_CH0_256, Weight_Gradient1_Layer8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
        Weight_Gradient2_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer8_CH0_256, Weight_Gradient2_Layer8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
        Weight_Gradient3_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer8_CH0_256, Weight_Gradient3_Layer8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
        Weight_Gradient4_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer8_CH0_256, Weight_Gradient4_Layer8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
        Weight_Gradient5_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer8_CH0_256, Weight_Gradient5_Layer8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
        Weight_Gradient6_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer8_CH0_256, Weight_Gradient6_Layer8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
        Weight_Gradient7_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer8_CH0_256, Weight_Gradient7_Layer8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
        Weight_Gradient8_Layer8 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer8_CH0_256, Weight_Gradient8_Layer8_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=1024, Layer8=True)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)
        
        Weight_Gradient_Layer8 = [Weight_Gradient1_Layer8, Weight_Gradient2_Layer8, Weight_Gradient3_Layer8, Weight_Gradient4_Layer8, Weight_Gradient5_Layer8, 
                                Weight_Gradient6_Layer8, Weight_Gradient7_Layer8, Weight_Gradient8_Layer8]
        
        # if DEBUG2 : Save_File(Weight_Gradient_Layer8, "result/Layer_8_Backward_Weight_Gradient")

        # Weight_Gradient_Layer8 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer8)]
        Weight_Gradient_Layer8 = list(np.mean(np.array(Weight_Gradient_Layer8), axis=0))
  
        Weight_Gradient_Layer8 = torch.tensor([float(value) for value in Weight_Gradient_Layer8], dtype=torch.float32).reshape(125, 1024, 1, 1)   

        if DEBUG2 : Save_File(Weight_Gradient_Layer8, "result/Layer_8_Backward_Weight_Gradient")
        
        # Backward_Const_List[7] = backward_const_7
        # Average_Per_Channel_List[7] = avg_pc_7

        # Weight_Backward_Layer7 for Soft2Hardware
        s = time.time()
        Weight_Backward_Layer7 = Weight_Hardware_Backward_ReOrdering_OtherLayer(1024, 1024, data.Weight_Bfloat[7], backward_const_7, avg_pc_7)
        e = time.time()
        if DEBUG: print("Weight Reordering : ",e-s)

        # Break 256To32 and Flip the Data: 
        s = time.time()
        Weight_Backward_Layer7_CH0 = data_256_32(Weight_Backward_Layer7[0])
        Weight_Backward_Layer7_CH1 = data_256_32(Weight_Backward_Layer7[1])
        e = time.time()
        if DEBUG: print("256bit to 32bit : ",e-s)

        # Write Weight For Backward into DDR
        s = time.time()
        Write_DDR(Weight_Backward_Layer7_CH0,Wr_Address=0x81340000)
        Write_DDR(Weight_Backward_Layer7_CH1,Wr_Address=0x91340000)
        e = time.time()
        if DEBUG: print("Write DDR : ",e-s)

        Blayer8_end = time.time()
        if DEBUG: print("Layer8 Process Time : ", Blayer8_end-Blayer8_start)


        resume()
        ##if DEBUG: print(irq_val)

        #################################################
        #             Backward Layer 7 Start            #
        #################################################
        layer7_start = time.time()
        # check Layer7 IRQ
        check_irq_otherlayer()
        # self.app_instance .change_color(self.app_instance.L8_IRQ_canvas, self.app_instance.L8_IRQ, "green")
        s = time.time()
        # Read Gradient of Output After ReLU Backward: 
        Output_Grad1_Layer7_CH0 = Read_DDR(Rd_Address=0x86B18000,  End_Address=0x86B4C000)
        Output_Grad1_Layer7_CH0_16 = data_32_to_16(Output_Grad1_Layer7_CH0)

        Output_Grad1_Layer7_CH1 = Read_DDR(Rd_Address=0x96B18000,  End_Address=0x96B4C000)
        Output_Grad1_Layer7_CH1_16 = data_32_to_16(Output_Grad1_Layer7_CH1)

        Output_Grad2_Layer7_CH0 = Read_DDR(Rd_Address=0x86B4C000,  End_Address=0x86B80000)
        Output_Grad2_Layer7_CH0_16 = data_32_to_16(Output_Grad2_Layer7_CH0)

        Output_Grad2_Layer7_CH1 = Read_DDR(Rd_Address=0x96B4C000,  End_Address=0x96B80000)
        Output_Grad2_Layer7_CH1_16 = data_32_to_16(Output_Grad2_Layer7_CH1)

        Output_Grad3_Layer7_CH0 = Read_DDR(Rd_Address=0x86B80000,  End_Address=0x86BB4000)
        Output_Grad3_Layer7_CH0_16 = data_32_to_16(Output_Grad3_Layer7_CH0)

        Output_Grad3_Layer7_CH1 = Read_DDR(Rd_Address=0x96B80000,  End_Address=0x96BB4000)
        Output_Grad3_Layer7_CH1_16 = data_32_to_16(Output_Grad3_Layer7_CH1)

        Output_Grad4_Layer7_CH0 = Read_DDR(Rd_Address=0x86BB4000,  End_Address=0x86BE8000)
        Output_Grad4_Layer7_CH0_16 = data_32_to_16(Output_Grad4_Layer7_CH0)

        Output_Grad4_Layer7_CH1 = Read_DDR(Rd_Address=0x96BB4000,  End_Address=0x96BE8000)
        Output_Grad4_Layer7_CH1_16 = data_32_to_16(Output_Grad4_Layer7_CH1)

        Output_Grad5_Layer7_CH0 = Read_DDR(Rd_Address=0x86BE8000,  End_Address=0x86C1C000)
        Output_Grad5_Layer7_CH0_16 = data_32_to_16(Output_Grad5_Layer7_CH0)

        Output_Grad5_Layer7_CH1 = Read_DDR(Rd_Address=0x96BE8000,  End_Address=0x96C1C000)
        Output_Grad5_Layer7_CH1_16 = data_32_to_16(Output_Grad5_Layer7_CH1)

        Output_Grad6_Layer7_CH0 = Read_DDR(Rd_Address=0x86C1C000,  End_Address=0x86C50000)
        Output_Grad6_Layer7_CH0_16 = data_32_to_16(Output_Grad6_Layer7_CH0)

        Output_Grad6_Layer7_CH1 = Read_DDR(Rd_Address=0x96C1C000,  End_Address=0x96C50000)
        Output_Grad6_Layer7_CH1_16 = data_32_to_16(Output_Grad6_Layer7_CH1)

        Output_Grad7_Layer7_CH0 = Read_DDR(Rd_Address=0x86C50000,  End_Address=0x86C84000)
        Output_Grad7_Layer7_CH0_16 = data_32_to_16(Output_Grad7_Layer7_CH0)

        Output_Grad7_Layer7_CH1 = Read_DDR(Rd_Address=0x96C50000,  End_Address=0x96C84000)
        Output_Grad7_Layer7_CH1_16 = data_32_to_16(Output_Grad7_Layer7_CH1)

        Output_Grad8_Layer7_CH0 = Read_DDR(Rd_Address=0x86C84000,  End_Address=0x86CB8000)
        Output_Grad8_Layer7_CH0_16 = data_32_to_16(Output_Grad8_Layer7_CH0)

        Output_Grad8_Layer7_CH1 = Read_DDR(Rd_Address=0x96C84000,  End_Address=0x96CB8000)
        Output_Grad8_Layer7_CH1_16 = data_32_to_16(Output_Grad8_Layer7_CH1)
        e = time.time()
        if DEBUG: print("Read Output_Gradient Time : ",e-s)

        s = time.time()
        Output_Grad1_Layer7 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer7_CH0_16, Output_Grad1_Layer7_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad2_Layer7 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer7_CH0_16, Output_Grad2_Layer7_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad3_Layer7 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer7_CH0_16, Output_Grad3_Layer7_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad4_Layer7 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer7_CH0_16, Output_Grad4_Layer7_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad5_Layer7 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer7_CH0_16, Output_Grad5_Layer7_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad6_Layer7 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer7_CH0_16, Output_Grad6_Layer7_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad7_Layer7 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer7_CH0_16, Output_Grad7_Layer7_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grad8_Layer7 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer7_CH0_16, Output_Grad8_Layer7_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        Output_Grads_Layer7 = Output_Grad1_Layer7 + Output_Grad2_Layer7 + Output_Grad3_Layer7 + Output_Grad4_Layer7 + \
                                Output_Grad5_Layer7 + Output_Grad6_Layer7 + Output_Grad7_Layer7 + Output_Grad8_Layer7    
        Output_Grad_Layer7 = torch.tensor([float(value) for value in Output_Grads_Layer7], dtype=torch.float32).reshape(8, 1024, 13, 13)
        e = time.time()
        if DEBUG: print("Read_OutFmap_Bfloat2Dec Time : ", e-s)
        if DEBUG2 : Save_File(Output_Grad_Layer7, "result/Layer_7_Backward_Input_Gradient")

        # BReLu Marking
        s = time.time()
        ReLu_Marking1_Layer6_CH0 = Read_DDR(Rd_Address=0x87F34000,  End_Address=0x87F68000)
        ReLu_Marking1_Layer6_CH0_256 = data_32_to_16(ReLu_Marking1_Layer6_CH0)

        ReLu_Marking2_Layer6_CH0 = Read_DDR(Rd_Address=0x87F68000,  End_Address=0x87F9C000)
        ReLu_Marking2_Layer6_CH0_256 = data_32_to_16(ReLu_Marking2_Layer6_CH0)

        ReLu_Marking3_Layer6_CH0 = Read_DDR(Rd_Address=0x87F9C000,  End_Address=0x87FD0000)
        ReLu_Marking3_Layer6_CH0_256 = data_32_to_16(ReLu_Marking3_Layer6_CH0)

        ReLu_Marking4_Layer6_CH0 = Read_DDR(Rd_Address=0x87FD0000,  End_Address=0x88004000)
        ReLu_Marking4_Layer6_CH0_256 = data_32_to_16(ReLu_Marking4_Layer6_CH0)

        ReLu_Marking5_Layer6_CH0 = Read_DDR(Rd_Address=0x88004000,  End_Address=0x88038000)
        ReLu_Marking5_Layer6_CH0_256 = data_32_to_16(ReLu_Marking5_Layer6_CH0)

        ReLu_Marking6_Layer6_CH0 = Read_DDR(Rd_Address=0x88038000,  End_Address=0x8806C000)
        ReLu_Marking6_Layer6_CH0_256 = data_32_to_16(ReLu_Marking6_Layer6_CH0)

        ReLu_Marking7_Layer6_CH0 = Read_DDR(Rd_Address=0x8806C000,  End_Address=0x880A0000)
        ReLu_Marking7_Layer6_CH0_256 = data_32_to_16(ReLu_Marking7_Layer6_CH0)

        ReLu_Marking8_Layer6_CH0 = Read_DDR(Rd_Address=0x880A0000,  End_Address=0x880D4000)
        ReLu_Marking8_Layer6_CH0_256 = data_32_to_16(ReLu_Marking8_Layer6_CH0)

        ReLu_Marking1_Layer6_CH1 = Read_DDR(Rd_Address=0x97F34000,  End_Address=0x97F68000)
        ReLu_Marking1_Layer6_CH1_256 = data_32_to_16(ReLu_Marking1_Layer6_CH1)

        ReLu_Marking2_Layer6_CH1 = Read_DDR(Rd_Address=0x97F68000,  End_Address=0x97F9C000)
        ReLu_Marking2_Layer6_CH1_256 = data_32_to_16(ReLu_Marking2_Layer6_CH1)

        ReLu_Marking3_Layer6_CH1 = Read_DDR(Rd_Address=0x97F9C000,  End_Address=0x97FD0000)
        ReLu_Marking3_Layer6_CH1_256 = data_32_to_16(ReLu_Marking3_Layer6_CH1)

        ReLu_Marking4_Layer6_CH1 = Read_DDR(Rd_Address=0x97FD0000,  End_Address=0x98004000)
        ReLu_Marking4_Layer6_CH1_256 = data_32_to_16(ReLu_Marking4_Layer6_CH1)

        ReLu_Marking5_Layer6_CH1 = Read_DDR(Rd_Address=0x98004000,  End_Address=0x98038000)
        ReLu_Marking5_Layer6_CH1_256 = data_32_to_16(ReLu_Marking5_Layer6_CH1)

        ReLu_Marking6_Layer6_CH1 = Read_DDR(Rd_Address=0x98038000,  End_Address=0x9806C000)
        ReLu_Marking6_Layer6_CH1_256 = data_32_to_16(ReLu_Marking6_Layer6_CH1)

        ReLu_Marking7_Layer6_CH1 = Read_DDR(Rd_Address=0x9806C000,  End_Address=0x980A0000)
        ReLu_Marking7_Layer6_CH1_256 = data_32_to_16(ReLu_Marking7_Layer6_CH1)

        ReLu_Marking8_Layer6_CH1 = Read_DDR(Rd_Address=0x980A0000,  End_Address=0x980D4000)
        ReLu_Marking8_Layer6_CH1_256 = data_32_to_16(ReLu_Marking8_Layer6_CH1)
        e = time.time()
        if DEBUG: print("Read ReLu_Marking Time : ",e-s)
        # ReLu Reordering
        s = time.time()
        ReLu_Marking1_Layer6 = Read_ReLu_Marking(ReLu_Marking1_Layer6_CH0_256, ReLu_Marking1_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        ReLu_Marking2_Layer6 = Read_ReLu_Marking(ReLu_Marking2_Layer6_CH0_256, ReLu_Marking2_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        ReLu_Marking3_Layer6 = Read_ReLu_Marking(ReLu_Marking3_Layer6_CH0_256, ReLu_Marking3_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        ReLu_Marking4_Layer6 = Read_ReLu_Marking(ReLu_Marking4_Layer6_CH0_256, ReLu_Marking4_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        ReLu_Marking5_Layer6 = Read_ReLu_Marking(ReLu_Marking5_Layer6_CH0_256, ReLu_Marking5_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        ReLu_Marking6_Layer6 = Read_ReLu_Marking(ReLu_Marking6_Layer6_CH0_256, ReLu_Marking6_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        ReLu_Marking7_Layer6 = Read_ReLu_Marking(ReLu_Marking7_Layer6_CH0_256, ReLu_Marking7_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)
        ReLu_Marking8_Layer6 = Read_ReLu_Marking(ReLu_Marking8_Layer6_CH0_256, ReLu_Marking8_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, Out_Size=13, Layer8=False)


        ReLu_Marking_Layer6 = ReLu_Marking1_Layer6 + ReLu_Marking2_Layer6 + ReLu_Marking3_Layer6 + ReLu_Marking4_Layer6 + ReLu_Marking5_Layer6 + \
                                ReLu_Marking6_Layer6 + ReLu_Marking7_Layer6 + ReLu_Marking8_Layer6
        
        ReLu_Marking_Layer6 = torch.tensor([float(value) for value in ReLu_Marking_Layer6], dtype=torch.float32).reshape(8, 1024, 13, 13)

        e = time.time()
        if DEBUG: print("ReLu Marking Convert Time : ",e-s)

        # BReLu Calculate
        # Output_Grad_layer7_input = torch.tensor(Output_Grad_Layer7, dtype=torch.float32).reshape(8,1024,13,13)
        # Layer6_Location = torch.tensor(ReLu_Marking_Layer6, dtype=torch.float32).reshape(8,1024,13,13)
        s = time.time()
        relu_mask, location_mask = split_location(ReLu_Marking_Layer6)
        grad_relu_output = backward_active(Output_Grad_Layer7, relu_mask)
        #grad_maxpool_output = backward_MaxPool_Location(grad_relu_output, location_mask)
        dL_dgamma_6, dL_dbeta_6, avg_pc_6, backward_const_6 = backward_LightNorm(grad_relu_output, layer6_cache)

        # avg_pc_6 = avg_pc_6.squeeze()
        # backward_const_6 = backward_const_6.squeeze()
        e = time.time()
        if DEBUG: print("Backward ReLu & Backward Maxpoolin Time : ",e-s)

        s = time.time()
        avg_pc_6, backward_const_6 = Mean_Var_Dec2Bfloat(avg_pc_6, backward_const_6, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Mean_Var_Dec2Bfloat Time : ",e-s)

        # Weight_Backward_Layer6 for Soft2Hardware
        s = time.time()
        Weight_Backward_Layer6 = Weight_Hardware_Backward_ReOrdering_OtherLayer(1024, 512, data.Weight_Bfloat[6], backward_const_6, avg_pc_6)
        e = time.time()
        if DEBUG: print("Weight_Hardware_Backward_ReOrdering_OtherLayer Time : ",e-s)


        # Break 256To32 and Flip the Data: 
        s = time.time()
        Weight_Backward_Layer6_CH0 = data_256_32(Weight_Backward_Layer6[0])
        Weight_Backward_Layer6_CH1 = data_256_32(Weight_Backward_Layer6[1])
        e = time.time()
        if DEBUG: print("data_256_32 Time : ",e-s)
        # Write Weight For Backward into DDR
        s = time.time()
        Write_DDR(Weight_Backward_Layer6_CH0,Wr_Address=0x81D40000)
        Write_DDR(Weight_Backward_Layer6_CH1,Wr_Address=0x91D40000)
        e = time.time()
        if DEBUG: print("Write_DDR Time : ",e-s)

        # Gradient of Beta Calculation:
        # Beta_Gradient_Layer7 = (Output_Grad_Layer7).sum(dim=(0, 2, 3), keepdim=True)

        # Weight Gradient
        s = time.time()
        Weight_Gradient1_Layer7_CH0 = Read_DDR(Rd_Address=0x883DC000,  End_Address=0x88CDC000)
        Weight_Gradient1_Layer7_CH0_256 = data_32_to_16(Weight_Gradient1_Layer7_CH0)
        #if DEBUG: print("Weight_Gradient1_Layer7_CH0 : ", len(Weight_Gradient1_Layer7_CH0))   

        Weight_Gradient2_Layer7_CH0 = Read_DDR(Rd_Address=0x88CDC000,  End_Address=0x895DC000)
        Weight_Gradient2_Layer7_CH0_256 = data_32_to_16(Weight_Gradient2_Layer7_CH0)
        #if DEBUG: print("Weight_Gradient2_Layer7_CH0 : ", len(Weight_Gradient2_Layer7_CH0))    

        Weight_Gradient3_Layer7_CH0 = Read_DDR(Rd_Address=0x895DC000,  End_Address=0x89EDC000)
        Weight_Gradient3_Layer7_CH0_256 = data_32_to_16(Weight_Gradient3_Layer7_CH0)
        #if DEBUG: print("Weight_Gradient3_Layer7_CH0 : ", len(Weight_Gradient3_Layer7_CH0)) 

        Weight_Gradient4_Layer7_CH0 = Read_DDR(Rd_Address=0x89EDC000,  End_Address=0x8A7DC000)
        Weight_Gradient4_Layer7_CH0_256 = data_32_to_16(Weight_Gradient4_Layer7_CH0)
        #if DEBUG: print("Weight_Gradient4_Layer7_CH0 : ", len(Weight_Gradient4_Layer7_CH0)) 

        Weight_Gradient5_Layer7_CH0 = Read_DDR(Rd_Address=0x8A7DC000,  End_Address=0x8B0DC000)
        Weight_Gradient5_Layer7_CH0_256 = data_32_to_16(Weight_Gradient5_Layer7_CH0)
        #if DEBUG: print("Weight_Gradient5_Layer7_CH0 : ", len(Weight_Gradient5_Layer7_CH0)) 

        Weight_Gradient6_Layer7_CH0 = Read_DDR(Rd_Address=0x8B0DC000,  End_Address=0x8B9DC000)
        Weight_Gradient6_Layer7_CH0_256 = data_32_to_16(Weight_Gradient6_Layer7_CH0)
        #if DEBUG: print("Weight_Gradient6_Layer7_CH0 : ", len(Weight_Gradient6_Layer7_CH0)) 

        Weight_Gradient7_Layer7_CH0 = Read_DDR(Rd_Address=0x8B9DC000,  End_Address=0x8C2DC000)
        Weight_Gradient7_Layer7_CH0_256 = data_32_to_16(Weight_Gradient7_Layer7_CH0)
        #if DEBUG: print("Weight_Gradient7_Layer7_CH0 : ", len(Weight_Gradient7_Layer7_CH0)) 

        Weight_Gradient8_Layer7_CH0 = Read_DDR(Rd_Address=0x8C2DC000,  End_Address=0x8CBDC000)
        Weight_Gradient8_Layer7_CH0_256 = data_32_to_16(Weight_Gradient8_Layer7_CH0)
        #if DEBUG: print("Weight_Gradient8_Layer7_CH0 : ", len(Weight_Gradient8_Layer7_CH0)) 

        Weight_Gradient1_Layer7_CH1 = Read_DDR(Rd_Address=0x983DC000,  End_Address=0x98CDC000)
        Weight_Gradient1_Layer7_CH1_256 = data_32_to_16(Weight_Gradient1_Layer7_CH1)
        #if DEBUG: print("Weight_Gradient1_Layer7_CH1 : ", len(Weight_Gradient1_Layer7_CH1)) 

        Weight_Gradient2_Layer7_CH1 = Read_DDR(Rd_Address=0x98CDC000,  End_Address=0x995DC000)
        Weight_Gradient2_Layer7_CH1_256 = data_32_to_16(Weight_Gradient2_Layer7_CH1)
        #if DEBUG: print("Weight_Gradient2_Layer7_CH1 : ", len(Weight_Gradient2_Layer7_CH1)) 

        Weight_Gradient3_Layer7_CH1 = Read_DDR(Rd_Address=0x995DC000,  End_Address=0x99EDC000)
        Weight_Gradient3_Layer7_CH1_256 = data_32_to_16(Weight_Gradient3_Layer7_CH1)
        #if DEBUG: print("Weight_Gradient3_Layer7_CH1 : ", len(Weight_Gradient3_Layer7_CH1)) 

        Weight_Gradient4_Layer7_CH1 = Read_DDR(Rd_Address=0x99EDC000,  End_Address=0x9A7DC000)
        Weight_Gradient4_Layer7_CH1_256 = data_32_to_16(Weight_Gradient4_Layer7_CH1)
        #if DEBUG: print("Weight_Gradient4_Layer7_CH1 : ", len(Weight_Gradient4_Layer7_CH1)) 

        Weight_Gradient5_Layer7_CH1 = Read_DDR(Rd_Address=0x9A7DC000,  End_Address=0x9B0DC000)
        Weight_Gradient5_Layer7_CH1_256 = data_32_to_16(Weight_Gradient5_Layer7_CH1)
        #if DEBUG: print("Weight_Gradient5_Layer7_CH1 : ", len(Weight_Gradient5_Layer7_CH1)) 

        Weight_Gradient6_Layer7_CH1 = Read_DDR(Rd_Address=0x9B0DC000,  End_Address=0x9B9DC000)
        Weight_Gradient6_Layer7_CH1_256 = data_32_to_16(Weight_Gradient6_Layer7_CH1)
        #if DEBUG: print("Weight_Gradient6_Layer7_CH1 : ", len(Weight_Gradient6_Layer7_CH1)) 

        Weight_Gradient7_Layer7_CH1 = Read_DDR(Rd_Address=0x9B9DC000,  End_Address=0x9C2DC000)
        Weight_Gradient7_Layer7_CH1_256 = data_32_to_16(Weight_Gradient7_Layer7_CH1)
        #if DEBUG: print("Weight_Gradient7_Layer7_CH1 : ", len(Weight_Gradient7_Layer7_CH1)) 

        Weight_Gradient8_Layer7_CH1 = Read_DDR(Rd_Address=0x9C2DC000,  End_Address=0x9CBDC000)
        Weight_Gradient8_Layer7_CH1_256 = data_32_to_16(Weight_Gradient8_Layer7_CH1)
        #if DEBUG: print("Weight_Gradient8_Layer7_CH1 : ", len(Weight_Gradient8_Layer7_CH1)) 
        e = time.time()
        if DEBUG: print("Read Weight_Gradient8 Time : ",e-s)

        '''
        test_out = 'Weight_Result/Weight_Gradient1_Layer7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient1_Layer7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer7_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer7_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer7_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer7_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''

        # Weight_Gradient_Layer7_CH0 = [Weight_Gradient1_Layer7_CH0_256, Weight_Gradient2_Layer7_CH0_256, Weight_Gradient3_Layer7_CH0_256, Weight_Gradient4_Layer7_CH0_256, 
        #                         Weight_Gradient5_Layer7_CH0_256, Weight_Gradient6_Layer7_CH0_256, Weight_Gradient7_Layer7_CH0_256, Weight_Gradient8_Layer7_CH0_256]
        
        # Weight_Gradient_Layer7_CH1 = [Weight_Gradient1_Layer7_CH1_256, Weight_Gradient2_Layer7_CH1_256, Weight_Gradient3_Layer7_CH1_256, Weight_Gradient4_Layer7_CH1_256, 
        #                         Weight_Gradient5_Layer7_CH1_256, Weight_Gradient6_Layer7_CH1_256, Weight_Gradient7_Layer7_CH1_256, Weight_Gradient8_Layer7_CH1_256]
        
        # Weight_Gradient_Layer7 = Read_WeightGradient_Bfloat2Dec_whole_image(Weight_Gradient_Layer7_CH0, Weight_Gradient_Layer7_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
        s = time.time()
        Weight_Gradient1_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer7_CH0_256, Weight_Gradient1_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
        Weight_Gradient2_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer7_CH0_256, Weight_Gradient2_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
        Weight_Gradient3_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer7_CH0_256, Weight_Gradient3_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
        Weight_Gradient4_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer7_CH0_256, Weight_Gradient4_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
        Weight_Gradient5_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer7_CH0_256, Weight_Gradient5_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
        Weight_Gradient6_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer7_CH0_256, Weight_Gradient6_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
        Weight_Gradient7_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer7_CH0_256, Weight_Gradient7_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
        Weight_Gradient8_Layer7 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer7_CH0_256, Weight_Gradient8_Layer7_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=1024, Layer8=False)
        Weight_Gradient_Layer7 = [Weight_Gradient1_Layer7, Weight_Gradient2_Layer7, Weight_Gradient3_Layer7, Weight_Gradient4_Layer7, Weight_Gradient5_Layer7, 
                                Weight_Gradient6_Layer7, Weight_Gradient7_Layer7, Weight_Gradient8_Layer7]
        
        # Weight_Gradient_Layer7 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer7)]   
        Weight_Gradient_Layer7 = list(np.mean(np.array(Weight_Gradient_Layer7), axis=0))
        
        Weight_Gradient_Layer7 = torch.tensor([float(value) for value in Weight_Gradient_Layer7], dtype=torch.float32).reshape(1024, 1024, 3, 3)  
        e = time.time()
        if DEBUG: print("WeightGradient_Bfloat2Dec Time : ",e-s)
        layer7_end = time.time()
        process_time = layer7_end - layer7_start
        if DEBUG: print("Layer7 Process Time : ", process_time)  

        resume()


        #################################################
        #             Backward Layer 6 Start            #
        #################################################
        layer6_start = time.time()
        # check Layer7 IRQ
        check_irq_otherlayer()
        s = time.time()
        # self.app_instance .change_color(self.app_instance.L7_IRQ_canvas, self.app_instance.L7_IRQ, "green")
        # Read Gradient of Output After ReLU Backward: 
        Output_Grad1_Layer6_CH0 = Read_DDR(Rd_Address=0x86978000,  End_Address=0x86992000)
        Output_Grad1_Layer6_CH0_16 = data_32_to_16(Output_Grad1_Layer6_CH0)

        Output_Grad1_Layer6_CH1 = Read_DDR(Rd_Address=0x96978000,  End_Address=0x96992000)
        Output_Grad1_Layer6_CH1_16 = data_32_to_16(Output_Grad1_Layer6_CH1)

        Output_Grad2_Layer6_CH0 = Read_DDR(Rd_Address=0x86992000,  End_Address=0x869AC000)
        Output_Grad2_Layer6_CH0_16 = data_32_to_16(Output_Grad2_Layer6_CH0)

        Output_Grad2_Layer6_CH1 = Read_DDR(Rd_Address=0x96992000,  End_Address=0x969AC000)
        Output_Grad2_Layer6_CH1_16 = data_32_to_16(Output_Grad2_Layer6_CH1)

        Output_Grad3_Layer6_CH0 = Read_DDR(Rd_Address=0x869AC000,  End_Address=0x869C6000)
        Output_Grad3_Layer6_CH0_16 = data_32_to_16(Output_Grad3_Layer6_CH0)

        Output_Grad3_Layer6_CH1 = Read_DDR(Rd_Address=0x969AC000,  End_Address=0x969C6000)
        Output_Grad3_Layer6_CH1_16 = data_32_to_16(Output_Grad3_Layer6_CH1)

        Output_Grad4_Layer6_CH0 = Read_DDR(Rd_Address=0x869C6000,  End_Address=0x869E0000)
        Output_Grad4_Layer6_CH0_16 = data_32_to_16(Output_Grad4_Layer6_CH0)

        Output_Grad4_Layer6_CH1 = Read_DDR(Rd_Address=0x969C6000,  End_Address=0x969E0000)
        Output_Grad4_Layer6_CH1_16 = data_32_to_16(Output_Grad4_Layer6_CH1)

        Output_Grad5_Layer6_CH0 = Read_DDR(Rd_Address=0x869E0000,  End_Address=0x869FA000)
        Output_Grad5_Layer6_CH0_16 = data_32_to_16(Output_Grad5_Layer6_CH0)

        Output_Grad5_Layer6_CH1 = Read_DDR(Rd_Address=0x969E0000,  End_Address=0x969FA000)
        Output_Grad5_Layer6_CH1_16 = data_32_to_16(Output_Grad5_Layer6_CH1)

        Output_Grad6_Layer6_CH0 = Read_DDR(Rd_Address=0x869FA000,  End_Address=0x86A14000)
        Output_Grad6_Layer6_CH0_16 = data_32_to_16(Output_Grad6_Layer6_CH0)

        Output_Grad6_Layer6_CH1 = Read_DDR(Rd_Address=0x969FA000,  End_Address=0x96A14000)
        Output_Grad6_Layer6_CH1_16 = data_32_to_16(Output_Grad6_Layer6_CH1)

        Output_Grad7_Layer6_CH0 = Read_DDR(Rd_Address=0x86A14000,  End_Address=0x86A2E000)
        Output_Grad7_Layer6_CH0_16 = data_32_to_16(Output_Grad7_Layer6_CH0)

        Output_Grad7_Layer6_CH1 = Read_DDR(Rd_Address=0x96A14000,  End_Address=0x96A2E000)
        Output_Grad7_Layer6_CH1_16 = data_32_to_16(Output_Grad7_Layer6_CH1)

        Output_Grad8_Layer6_CH0 = Read_DDR(Rd_Address=0x86A2E000,  End_Address=0x86A48000)
        Output_Grad8_Layer6_CH0_16 = data_32_to_16(Output_Grad8_Layer6_CH0)

        Output_Grad8_Layer6_CH1 = Read_DDR(Rd_Address=0x96A2E000,  End_Address=0x96A48000)
        Output_Grad8_Layer6_CH1_16 = data_32_to_16(Output_Grad8_Layer6_CH1)
        e = time.time()
        if DEBUG: print("Read OG DDR & 32bit to 16bit : ",e-s)

        s = time.time()
        Output_Grad1_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer6_CH0_16, Output_Grad1_Layer6_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Grad2_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer6_CH0_16, Output_Grad2_Layer6_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Grad3_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer6_CH0_16, Output_Grad3_Layer6_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Grad4_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer6_CH0_16, Output_Grad4_Layer6_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Grad5_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer6_CH0_16, Output_Grad5_Layer6_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Grad6_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer6_CH0_16, Output_Grad6_Layer6_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Grad7_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer6_CH0_16, Output_Grad7_Layer6_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        Output_Grad8_Layer6 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer6_CH0_16, Output_Grad8_Layer6_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)
        
        Output_Grads_Layer6 = Output_Grad1_Layer6 + Output_Grad2_Layer6 + Output_Grad3_Layer6 + Output_Grad4_Layer6 + \
                                Output_Grad5_Layer6 + Output_Grad6_Layer6 + Output_Grad7_Layer6 + Output_Grad8_Layer6    
        Output_Grad_Layer6 = torch.tensor([float(value) for value in Output_Grads_Layer6], dtype=torch.float32).reshape(8, 512, 13, 13)


        # BReLu Marking
        s = time.time()
        ReLu_Marking1_Layer5_CH0 = Read_DDR(Rd_Address=0x87E64000,  End_Address=0x87E7E000)
        ReLu_Marking1_Layer5_CH0_256 = data_32_to_16(ReLu_Marking1_Layer5_CH0)

        ReLu_Marking2_Layer5_CH0 = Read_DDR(Rd_Address=0x87E7E000,  End_Address=0x87E98000)
        ReLu_Marking2_Layer5_CH0_256 = data_32_to_16(ReLu_Marking2_Layer5_CH0)

        ReLu_Marking3_Layer5_CH0 = Read_DDR(Rd_Address=0x87E98000,  End_Address=0x87EB2000)
        ReLu_Marking3_Layer5_CH0_256 = data_32_to_16(ReLu_Marking3_Layer5_CH0)

        ReLu_Marking4_Layer5_CH0 = Read_DDR(Rd_Address=0x87EB2000,  End_Address=0x87ECC000)
        ReLu_Marking4_Layer5_CH0_256 = data_32_to_16(ReLu_Marking4_Layer5_CH0)

        ReLu_Marking5_Layer5_CH0 = Read_DDR(Rd_Address=0x87ECC000,  End_Address=0x87EE6000)
        ReLu_Marking5_Layer5_CH0_256 = data_32_to_16(ReLu_Marking5_Layer5_CH0)

        ReLu_Marking6_Layer5_CH0 = Read_DDR(Rd_Address=0x87EE6000,  End_Address=0x87F00000)
        ReLu_Marking6_Layer5_CH0_256 = data_32_to_16(ReLu_Marking6_Layer5_CH0)

        ReLu_Marking7_Layer5_CH0 = Read_DDR(Rd_Address=0x87F00000,  End_Address=0x87F1A000)
        ReLu_Marking7_Layer5_CH0_256 = data_32_to_16(ReLu_Marking7_Layer5_CH0)

        ReLu_Marking8_Layer5_CH0 = Read_DDR(Rd_Address=0x87F1A000,  End_Address=0x87F34000)
        ReLu_Marking8_Layer5_CH0_256 = data_32_to_16(ReLu_Marking8_Layer5_CH0)

        ReLu_Marking1_Layer5_CH1 = Read_DDR(Rd_Address=0x97E64000,  End_Address=0x97E7E000)
        ReLu_Marking1_Layer5_CH1_256 = data_32_to_16(ReLu_Marking1_Layer5_CH1)

        ReLu_Marking2_Layer5_CH1 = Read_DDR(Rd_Address=0x97E7E000,  End_Address=0x97E98000)
        ReLu_Marking2_Layer5_CH1_256 = data_32_to_16(ReLu_Marking2_Layer5_CH1)

        ReLu_Marking3_Layer5_CH1 = Read_DDR(Rd_Address=0x97E98000,  End_Address=0x97EB2000)
        ReLu_Marking3_Layer5_CH1_256 = data_32_to_16(ReLu_Marking3_Layer5_CH1)

        ReLu_Marking4_Layer5_CH1 = Read_DDR(Rd_Address=0x97EB2000,  End_Address=0x97ECC000)
        ReLu_Marking4_Layer5_CH1_256 = data_32_to_16(ReLu_Marking4_Layer5_CH1)

        ReLu_Marking5_Layer5_CH1 = Read_DDR(Rd_Address=0x97ECC000,  End_Address=0x97EE6000)
        ReLu_Marking5_Layer5_CH1_256 = data_32_to_16(ReLu_Marking5_Layer5_CH1)

        ReLu_Marking6_Layer5_CH1 = Read_DDR(Rd_Address=0x97EE6000,  End_Address=0x97F00000)
        ReLu_Marking6_Layer5_CH1_256 = data_32_to_16(ReLu_Marking6_Layer5_CH1)

        ReLu_Marking7_Layer5_CH1 = Read_DDR(Rd_Address=0x97F00000,  End_Address=0x97F1A000)
        ReLu_Marking7_Layer5_CH1_256 = data_32_to_16(ReLu_Marking7_Layer5_CH1)

        ReLu_Marking8_Layer5_CH1 = Read_DDR(Rd_Address=0x97F1A000,  End_Address=0x97F34000)
        ReLu_Marking8_Layer5_CH1_256 = data_32_to_16(ReLu_Marking8_Layer5_CH1)
        e = time.time()
        if DEBUG: print("Read RM DDR & 32bit to 16bit : ",e-s)

        # ReLu Reordering
        s = time.time()
        ReLu_Marking1_Layer5 = Read_ReLu_Marking(ReLu_Marking1_Layer5_CH0_256, ReLu_Marking1_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        ReLu_Marking2_Layer5 = Read_ReLu_Marking(ReLu_Marking2_Layer5_CH0_256, ReLu_Marking2_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        ReLu_Marking3_Layer5 = Read_ReLu_Marking(ReLu_Marking3_Layer5_CH0_256, ReLu_Marking3_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        ReLu_Marking4_Layer5 = Read_ReLu_Marking(ReLu_Marking4_Layer5_CH0_256, ReLu_Marking4_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        ReLu_Marking5_Layer5 = Read_ReLu_Marking(ReLu_Marking5_Layer5_CH0_256, ReLu_Marking5_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        ReLu_Marking6_Layer5 = Read_ReLu_Marking(ReLu_Marking6_Layer5_CH0_256, ReLu_Marking6_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        ReLu_Marking7_Layer5 = Read_ReLu_Marking(ReLu_Marking7_Layer5_CH0_256, ReLu_Marking7_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        ReLu_Marking8_Layer5 = Read_ReLu_Marking(ReLu_Marking8_Layer5_CH0_256, ReLu_Marking8_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, Out_Size=13, Layer8=False)
        e = time.time()
        if DEBUG: print("ReLu Reordering : ",e-s)

        ReLu_Marking_Layer5 = ReLu_Marking1_Layer5 + ReLu_Marking2_Layer5 + ReLu_Marking3_Layer5 + ReLu_Marking4_Layer5 + ReLu_Marking5_Layer5 + \
                                ReLu_Marking6_Layer5 + ReLu_Marking7_Layer5 + ReLu_Marking8_Layer5
        
        ReLu_Marking_Layer5 = torch.tensor([float(value) for value in ReLu_Marking_Layer5], dtype=torch.float32).reshape(8, 512, 13, 13)


        # BReLu Calculate
        # Output_Grad_layer6_input = torch.tensor(Output_Grad_Layer6, dtype=torch.float32).reshape(8,512,13,13)
        # Layer5_Location = torch.tensor(ReLu_Marking_Layer5, dtype=torch.float32).reshape(8,512,13,13)
        s = time.time()
        relu_mask, location_mask = split_location(ReLu_Marking_Layer5)
        grad_relu_output = backward_active(Output_Grad_Layer6, relu_mask)
        #grad_maxpool_output = backward_MaxPool_Location(grad_relu_output, location_mask)
        dL_dgamma_5, dL_dbeta_5, avg_pc_5, backward_const_5 = backward_LightNorm(grad_relu_output, layer5_cache)
        e = time.time()
        if DEBUG: print("Software : ",e-s)

        # avg_pc_5 = avg_pc_5.squeeze()
        # backward_const_5 = backward_const_5.squeeze()
        s = time.time()
        avg_pc_5, backward_const_5 = Mean_Var_Dec2Bfloat(avg_pc_5, backward_const_5, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat : ",e-s)

        # Weight_Backward_Layer5 for Soft2Hardware
        s = time.time()
        Weight_Backward_Layer5 = Weight_Hardware_Backward_ReOrdering_OtherLayer(512, 256, data.Weight_Bfloat[5], backward_const_5, avg_pc_5)
        e = time.time()
        if DEBUG: print("Weight Reordering : ",e-s)

        # Break 256To32 and Flip the Data: 
        s = time.time()
        Weight_Backward_Layer5_CH0 = data_256_32(Weight_Backward_Layer5[0])
        Weight_Backward_Layer5_CH1 = data_256_32(Weight_Backward_Layer5[1])
        e = time.time()
        if DEBUG: print("256bit to 32bit : ",e-s)

        # Write Weight For Backward into DDR
        s = time.time()
        Write_DDR(Weight_Backward_Layer5_CH0,Wr_Address=0x82240000)
        Write_DDR(Weight_Backward_Layer5_CH1,Wr_Address=0x92240000)
        e = time.time()
        if DEBUG: print("Write DDR : ",e-s)

        
        # Gradient of Beta Calculation:
        # Beta_Gradient_Layer6 = (Output_Grad_Layer6).sum(dim=(0, 2, 3), keepdim=True)

        # Weight Gradient
        s = time.time()
        Weight_Gradient1_Layer6_CH0 = Read_DDR(Rd_Address=0x8CBDC000,  End_Address=0x8D05C000)
        Weight_Gradient1_Layer6_CH0_256 = data_32_to_16(Weight_Gradient1_Layer6_CH0)
        #if DEBUG: print("Weight_Gradient1_Layer6_CH0 : ", len(Weight_Gradient1_Layer6_CH0))   

        Weight_Gradient2_Layer6_CH0 = Read_DDR(Rd_Address=0x8D05C000,  End_Address=0x8D4DC000)
        Weight_Gradient2_Layer6_CH0_256 = data_32_to_16(Weight_Gradient2_Layer6_CH0)
        #if DEBUG: print("Weight_Gradient2_Layer6_CH0 : ", len(Weight_Gradient2_Layer6_CH0))    

        Weight_Gradient3_Layer6_CH0 = Read_DDR(Rd_Address=0x8D4DC000,  End_Address=0x8D95C000)
        Weight_Gradient3_Layer6_CH0_256 = data_32_to_16(Weight_Gradient3_Layer6_CH0)
        #if DEBUG: print("Weight_Gradient3_Layer6_CH0 : ", len(Weight_Gradient3_Layer6_CH0)) 

        Weight_Gradient4_Layer6_CH0 = Read_DDR(Rd_Address=0x8D95C000,  End_Address=0x8DDDC000)
        Weight_Gradient4_Layer6_CH0_256 = data_32_to_16(Weight_Gradient4_Layer6_CH0)
        #if DEBUG: print("Weight_Gradient4_Layer6_CH0 : ", len(Weight_Gradient4_Layer6_CH0)) 

        Weight_Gradient5_Layer6_CH0 = Read_DDR(Rd_Address=0x8DDDC000,  End_Address=0x8E25C000)
        Weight_Gradient5_Layer6_CH0_256 = data_32_to_16(Weight_Gradient5_Layer6_CH0)
        #if DEBUG: print("Weight_Gradient5_Layer6_CH0 : ", len(Weight_Gradient5_Layer6_CH0)) 

        Weight_Gradient6_Layer6_CH0 = Read_DDR(Rd_Address=0x8E25C000,  End_Address=0x8E6DC000)
        Weight_Gradient6_Layer6_CH0_256 = data_32_to_16(Weight_Gradient6_Layer6_CH0)
        #if DEBUG: print("Weight_Gradient6_Layer6_CH0 : ", len(Weight_Gradient6_Layer6_CH0)) 

        Weight_Gradient7_Layer6_CH0 = Read_DDR(Rd_Address=0x8E6DC000,  End_Address=0x8EB5C000)
        Weight_Gradient7_Layer6_CH0_256 = data_32_to_16(Weight_Gradient7_Layer6_CH0)
        #if DEBUG: print("Weight_Gradient7_Layer6_CH0 : ", len(Weight_Gradient7_Layer6_CH0)) 

        Weight_Gradient8_Layer6_CH0 = Read_DDR(Rd_Address=0x8EB5C000,  End_Address=0x8EFDC000)
        Weight_Gradient8_Layer6_CH0_256 = data_32_to_16(Weight_Gradient8_Layer6_CH0)
        #if DEBUG: print("Weight_Gradient8_Layer6_CH0 : ", len(Weight_Gradient8_Layer6_CH0)) 

        Weight_Gradient1_Layer6_CH1 = Read_DDR(Rd_Address=0x9CBDC000,  End_Address=0x9D05C000)
        Weight_Gradient1_Layer6_CH1_256 = data_32_to_16(Weight_Gradient1_Layer6_CH1)
        #if DEBUG: print("Weight_Gradient1_Layer6_CH1 : ", len(Weight_Gradient1_Layer6_CH1)) 

        Weight_Gradient2_Layer6_CH1 = Read_DDR(Rd_Address=0x9D05C000,  End_Address=0x9D4DC000)
        Weight_Gradient2_Layer6_CH1_256 = data_32_to_16(Weight_Gradient2_Layer6_CH1)
        #if DEBUG: print("Weight_Gradient2_Layer6_CH1 : ", len(Weight_Gradient2_Layer6_CH1)) 

        Weight_Gradient3_Layer6_CH1 = Read_DDR(Rd_Address=0x9D4DC000,  End_Address=0x9D95C000)
        Weight_Gradient3_Layer6_CH1_256 = data_32_to_16(Weight_Gradient3_Layer6_CH1)
        #if DEBUG: print("Weight_Gradient3_Layer6_CH1 : ", len(Weight_Gradient3_Layer6_CH1)) 

        Weight_Gradient4_Layer6_CH1 = Read_DDR(Rd_Address=0x9D95C000,  End_Address=0x9DDDC000)
        Weight_Gradient4_Layer6_CH1_256 = data_32_to_16(Weight_Gradient4_Layer6_CH1)
        #if DEBUG: print("Weight_Gradient4_Layer6_CH1 : ", len(Weight_Gradient4_Layer6_CH1)) 

        Weight_Gradient5_Layer6_CH1 = Read_DDR(Rd_Address=0x9DDDC000,  End_Address=0x9E25C000)
        Weight_Gradient5_Layer6_CH1_256 = data_32_to_16(Weight_Gradient5_Layer6_CH1)
        #if DEBUG: print("Weight_Gradient5_Layer6_CH1 : ", len(Weight_Gradient5_Layer6_CH1)) 

        Weight_Gradient6_Layer6_CH1 = Read_DDR(Rd_Address=0x9E25C000,  End_Address=0x9E6DC000)
        Weight_Gradient6_Layer6_CH1_256 = data_32_to_16(Weight_Gradient6_Layer6_CH1)
        #if DEBUG: print("Weight_Gradient6_Layer6_CH1 : ", len(Weight_Gradient6_Layer6_CH1)) 

        Weight_Gradient7_Layer6_CH1 = Read_DDR(Rd_Address=0x9E6DC000,  End_Address=0x9EB5C000)
        Weight_Gradient7_Layer6_CH1_256 = data_32_to_16(Weight_Gradient7_Layer6_CH1)
        #if DEBUG: print("Weight_Gradient7_Layer6_CH1 : ", len(Weight_Gradient7_Layer6_CH1)) 

        Weight_Gradient8_Layer6_CH1 = Read_DDR(Rd_Address=0x9EB5C000,  End_Address=0x9EFDC000)
        Weight_Gradient8_Layer6_CH1_256 = data_32_to_16(Weight_Gradient8_Layer6_CH1)
        #if DEBUG: print("Weight_Gradient8_Layer6_CH1 : ", len(Weight_Gradient8_Layer6_CH1)) 
        e = time.time()
        if DEBUG: print("Read WG DDR & 32bit to 16bit : ",e-s)

        '''
        test_out = 'Weight_Result/Weight_Gradient1_Layer6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient1_Layer6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer6_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer6_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer6_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer6_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''

        s = time.time()
        Weight_Gradient1_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer6_CH0_256, Weight_Gradient1_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
        Weight_Gradient2_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer6_CH0_256, Weight_Gradient2_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
        Weight_Gradient3_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer6_CH0_256, Weight_Gradient3_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
        Weight_Gradient4_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer6_CH0_256, Weight_Gradient4_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
        Weight_Gradient5_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer6_CH0_256, Weight_Gradient5_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
        Weight_Gradient6_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer6_CH0_256, Weight_Gradient6_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
        Weight_Gradient7_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer6_CH0_256, Weight_Gradient7_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
        Weight_Gradient8_Layer6 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer6_CH0_256, Weight_Gradient8_Layer6_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=1024, In_CH=512, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)
        
        Weight_Gradient_Layer6 = [Weight_Gradient1_Layer6, Weight_Gradient2_Layer6, Weight_Gradient3_Layer6, Weight_Gradient4_Layer6, Weight_Gradient5_Layer6, 
                                Weight_Gradient6_Layer6, Weight_Gradient7_Layer6, Weight_Gradient8_Layer6]
        
        # Weight_Gradient_Layer6 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer6)]  
        Weight_Gradient_Layer6 = list(np.mean(np.array(Weight_Gradient_Layer6), axis=0))
        
         
        Weight_Gradient_Layer6 = torch.tensor([float(value) for value in Weight_Gradient_Layer6], dtype=torch.float32).reshape(1024, 512, 3, 3)  

        layer6_end = time.time()
        process_time = layer6_end - layer6_start
        if DEBUG: print("Layer6 Process Time : ", process_time)

        resume()

        #################################################
        #             Backward Layer 5 Start            #
        #################################################
        layer5_start = time.time()
        # check Layer5 IRQ
        check_irq_otherlayer()
        s = time.time()
        # self.app_instance .change_color(self.app_instance.L6_IRQ_canvas, self.app_instance.L6_IRQ, "green")
        # Read Gradient of Output After ReLU Backward: 
        Output_Grad1_Layer5_CH0_ = Read_DDR(Rd_Address=0x86770000,  End_Address=0x8677D000)
        Output_Grad1_Layer5_CH0_16 = data_32_to_16(Output_Grad1_Layer5_CH0_)
        #if DEBUG: print("Read Output_Grad1_Layer5_CH0")

        Output_Grad1_Layer5_CH1_ = Read_DDR(Rd_Address=0x96770000,  End_Address=0x9677D000)
        Output_Grad1_Layer5_CH1_16 = data_32_to_16(Output_Grad1_Layer5_CH1_)
        #if DEBUG: print("Read Output_Grad1_Layer5_CH1")

        Output_Grad2_Layer5_CH0_ = Read_DDR(Rd_Address=0x8677D000,  End_Address=0x8678A000)
        Output_Grad2_Layer5_CH0_16 = data_32_to_16(Output_Grad2_Layer5_CH0_)
        #if DEBUG: print("Read Output_Grad2_Layer5_CH0")

        Output_Grad2_Layer5_CH1_ = Read_DDR(Rd_Address=0x9677D000,  End_Address=0x9678A000)
        Output_Grad2_Layer5_CH1_16 = data_32_to_16(Output_Grad2_Layer5_CH1_)
        #if DEBUG: print("Read Output_Grad2_Layer5_CH1")

        Output_Grad3_Layer5_CH0_ = Read_DDR(Rd_Address=0x8678A000,  End_Address=0X86797000)
        Output_Grad3_Layer5_CH0_16 = data_32_to_16(Output_Grad3_Layer5_CH0_)
        #if DEBUG: print("Read Output_Grad3_Layer5_CH0")

        Output_Grad3_Layer5_CH1_ = Read_DDR(Rd_Address=0x9678A000,  End_Address=0x96797000)
        Output_Grad3_Layer5_CH1_16 = data_32_to_16(Output_Grad3_Layer5_CH1_)
        #if DEBUG: print("Read Output_Grad3_Layer5_CH1")

        Output_Grad4_Layer5_CH0_ = Read_DDR(Rd_Address=0x86797000,  End_Address=0x867A4000)
        Output_Grad4_Layer5_CH0_16 = data_32_to_16(Output_Grad4_Layer5_CH0_)
        #if DEBUG: print("Read Output_Grad4_Layer5_CH0")

        Output_Grad4_Layer5_CH1_ = Read_DDR(Rd_Address=0x96797000,  End_Address=0x967A4000)
        Output_Grad4_Layer5_CH1_16 = data_32_to_16(Output_Grad4_Layer5_CH1_)
        #if DEBUG: print("Read Output_Grad4_Layer5_CH1")

        Output_Grad5_Layer5_CH0_ = Read_DDR(Rd_Address=0x867A4000,  End_Address=0x867B1000)
        Output_Grad5_Layer5_CH0_16 = data_32_to_16(Output_Grad5_Layer5_CH0_)
        #if DEBUG: print("Read Output_Grad5_Layer5_CH0")

        Output_Grad5_Layer5_CH1_ = Read_DDR(Rd_Address=0x967A4000,  End_Address=0x967B1000)
        Output_Grad5_Layer5_CH1_16 = data_32_to_16(Output_Grad5_Layer5_CH1_)
        #if DEBUG: print("Read Output_Grad5_Layer5_CH1")

        Output_Grad6_Layer5_CH0_ = Read_DDR(Rd_Address=0x867B1000,  End_Address=0x867BE000)
        Output_Grad6_Layer5_CH0_16 = data_32_to_16(Output_Grad6_Layer5_CH0_)
        #if DEBUG: print("Read Output_Grad6_Layer5_CH0")

        Output_Grad6_Layer5_CH1_ = Read_DDR(Rd_Address=0x967B1000,  End_Address=0x967BE000)
        Output_Grad6_Layer5_CH1_16 = data_32_to_16(Output_Grad6_Layer5_CH1_)
        #if DEBUG: print("Read Output_Grad6_Layer5_CH1")

        Output_Grad7_Layer5_CH0_ = Read_DDR(Rd_Address=0x867BE000,  End_Address=0x867CB000)
        Output_Grad7_Layer5_CH0_16 = data_32_to_16(Output_Grad7_Layer5_CH0_)
        #if DEBUG: print("Read Output_Grad7_Layer5_CH0")

        Output_Grad7_Layer5_CH1_ = Read_DDR(Rd_Address=0x967BE000,  End_Address=0x967CB000)
        Output_Grad7_Layer5_CH1_16 = data_32_to_16(Output_Grad7_Layer5_CH1_)
        #if DEBUG: print("Read Output_Grad7_Layer5_CH1")

        Output_Grad8_Layer5_CH0_ = Read_DDR(Rd_Address=0x867CB000,  End_Address=0x867D8000)
        Output_Grad8_Layer5_CH0_16 = data_32_to_16(Output_Grad8_Layer5_CH0_)
        #if DEBUG: print("Read Output_Grad8_Layer5_CH0")

        Output_Grad8_Layer5_CH1_ = Read_DDR(Rd_Address=0x967CB000,  End_Address=0x967D8000)
        Output_Grad8_Layer5_CH1_16 = data_32_to_16(Output_Grad8_Layer5_CH1_)
        #if DEBUG: print("Read Output_Grad8_Layer5_CH1")
        e = time.time()
        if DEBUG: print("Read OG DDR & 32bit to 16bit : ",e-s)

        s = time.time()
        Output_Grad1_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer5_CH0_16, Output_Grad1_Layer5_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
        Output_Grad2_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer5_CH0_16, Output_Grad2_Layer5_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
        Output_Grad3_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer5_CH0_16, Output_Grad3_Layer5_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
        Output_Grad4_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer5_CH0_16, Output_Grad4_Layer5_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
        Output_Grad5_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer5_CH0_16, Output_Grad5_Layer5_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
        Output_Grad6_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer5_CH0_16, Output_Grad6_Layer5_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
        Output_Grad7_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer5_CH0_16, Output_Grad7_Layer5_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
        Output_Grad8_Layer5 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer5_CH0_16, Output_Grad8_Layer5_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)
        
        Output_Grads_Layer5 = Output_Grad1_Layer5 + Output_Grad2_Layer5 + Output_Grad3_Layer5 + Output_Grad4_Layer5 + \
                                Output_Grad5_Layer5 + Output_Grad6_Layer5 + Output_Grad7_Layer5 + Output_Grad8_Layer5    
        Output_Grad_Layer5 = torch.tensor([float(value) for value in Output_Grads_Layer5], dtype=torch.float32).reshape(8, 256, 13, 13)

        # ReLU
        s = time.time()
        ReLu_Marking1_Layer4_CH0 = Read_DDR(Rd_Address=0x87DFC000,  End_Address=0x87E09000)
        ReLu_Marking1_Layer4_CH0_256 = data_32_to_16(ReLu_Marking1_Layer4_CH0)

        ReLu_Marking2_Layer4_CH0 = Read_DDR(Rd_Address=0x87E09000,  End_Address=0x87E16000)
        ReLu_Marking2_Layer4_CH0_256 = data_32_to_16(ReLu_Marking2_Layer4_CH0)

        ReLu_Marking3_Layer4_CH0 = Read_DDR(Rd_Address=0x87E16000,  End_Address=0x87E23000)
        ReLu_Marking3_Layer4_CH0_256 = data_32_to_16(ReLu_Marking3_Layer4_CH0)

        ReLu_Marking4_Layer4_CH0 = Read_DDR(Rd_Address=0x87E23000,  End_Address=0x87E30000)
        ReLu_Marking4_Layer4_CH0_256 = data_32_to_16(ReLu_Marking4_Layer4_CH0)

        ReLu_Marking5_Layer4_CH0 = Read_DDR(Rd_Address=0x87E30000,  End_Address=0x87E3D000)
        ReLu_Marking5_Layer4_CH0_256 = data_32_to_16(ReLu_Marking5_Layer4_CH0)

        ReLu_Marking6_Layer4_CH0 = Read_DDR(Rd_Address=0x87E3D000,  End_Address=0x87E4A000)
        ReLu_Marking6_Layer4_CH0_256 = data_32_to_16(ReLu_Marking6_Layer4_CH0)

        ReLu_Marking7_Layer4_CH0 = Read_DDR(Rd_Address=0x87E4A000,  End_Address=0x87E57000)
        ReLu_Marking7_Layer4_CH0_256 = data_32_to_16(ReLu_Marking7_Layer4_CH0)

        ReLu_Marking8_Layer4_CH0 = Read_DDR(Rd_Address=0x87E57000,  End_Address=0x87E64000)
        ReLu_Marking8_Layer4_CH0_256 = data_32_to_16(ReLu_Marking8_Layer4_CH0)

        ReLu_Marking1_Layer4_CH1 = Read_DDR(Rd_Address=0x97DFC000,  End_Address=0x97E09000)
        ReLu_Marking1_Layer4_CH1_256 = data_32_to_16(ReLu_Marking1_Layer4_CH1)

        ReLu_Marking2_Layer4_CH1 = Read_DDR(Rd_Address=0x97E09000,  End_Address=0x97E16000)
        ReLu_Marking2_Layer4_CH1_256 = data_32_to_16(ReLu_Marking2_Layer4_CH1)

        ReLu_Marking3_Layer4_CH1 = Read_DDR(Rd_Address=0x97E16000,  End_Address=0x97E23000)
        ReLu_Marking3_Layer4_CH1_256 = data_32_to_16(ReLu_Marking3_Layer4_CH1)

        ReLu_Marking4_Layer4_CH1 = Read_DDR(Rd_Address=0x97E23000,  End_Address=0x97E30000)
        ReLu_Marking4_Layer4_CH1_256 = data_32_to_16(ReLu_Marking4_Layer4_CH1)

        ReLu_Marking5_Layer4_CH1 = Read_DDR(Rd_Address=0x97E30000,  End_Address=0x97E3D000)
        ReLu_Marking5_Layer4_CH1_256 = data_32_to_16(ReLu_Marking5_Layer4_CH1)

        ReLu_Marking6_Layer4_CH1 = Read_DDR(Rd_Address=0x97E3D000,  End_Address=0x97E4A000)
        ReLu_Marking6_Layer4_CH1_256 = data_32_to_16(ReLu_Marking6_Layer4_CH1)

        ReLu_Marking7_Layer4_CH1 = Read_DDR(Rd_Address=0x97E4A000,  End_Address=0x97E57000)
        ReLu_Marking7_Layer4_CH1_256 = data_32_to_16(ReLu_Marking7_Layer4_CH1)

        ReLu_Marking8_Layer4_CH1 = Read_DDR(Rd_Address=0x97E57000,  End_Address=0x97E64000)
        ReLu_Marking8_Layer4_CH1_256 = data_32_to_16(ReLu_Marking8_Layer4_CH1)
        e = time.time()
        if DEBUG: print("Read RM DDR & 32bit to 16bit : ",e-s)

        # ReLu Reordering
        s = time.time()
        ReLu_Marking1_Layer4 = Read_ReLu_Marking(ReLu_Marking1_Layer4_CH0_256, ReLu_Marking1_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
        ReLu_Marking2_Layer4 = Read_ReLu_Marking(ReLu_Marking2_Layer4_CH0_256, ReLu_Marking2_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
        ReLu_Marking3_Layer4 = Read_ReLu_Marking(ReLu_Marking3_Layer4_CH0_256, ReLu_Marking3_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
        ReLu_Marking4_Layer4 = Read_ReLu_Marking(ReLu_Marking4_Layer4_CH0_256, ReLu_Marking4_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
        ReLu_Marking5_Layer4 = Read_ReLu_Marking(ReLu_Marking5_Layer4_CH0_256, ReLu_Marking5_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
        ReLu_Marking6_Layer4 = Read_ReLu_Marking(ReLu_Marking6_Layer4_CH0_256, ReLu_Marking6_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
        ReLu_Marking7_Layer4 = Read_ReLu_Marking(ReLu_Marking7_Layer4_CH0_256, ReLu_Marking7_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
        ReLu_Marking8_Layer4 = Read_ReLu_Marking(ReLu_Marking8_Layer4_CH0_256, ReLu_Marking8_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, Out_Size=13, Layer8=False)
        e = time.time()
        if DEBUG: print("ReLu Convert : ",e-s)

        ReLu_Marking_Layer4 = ReLu_Marking1_Layer4 + ReLu_Marking2_Layer4 + ReLu_Marking3_Layer4 + ReLu_Marking4_Layer4 + ReLu_Marking5_Layer4 + \
                                ReLu_Marking6_Layer4 + ReLu_Marking7_Layer4 + ReLu_Marking8_Layer4
        
        ReLu_Marking_Layer4 = torch.tensor([float(value) for value in ReLu_Marking_Layer4], dtype=torch.float32).reshape(8, 256, 13, 13)

        # BReLu Calculate
        # Output_Grad_layer5_input = torch.tensor(Output_Grad_Layer5, dtype=torch.float32).reshape(8,256,13,13)
        # Layer4_Location = torch.tensor(ReLu_Marking_Layer4, dtype=torch.float32).reshape(8,256,13,13)

        s = time.time()
        relu_mask, location_mask = split_location(ReLu_Marking_Layer4)
        grad_relu_output = backward_active(Output_Grad_Layer5, relu_mask)
        grad_maxpool_output = backward_MaxPool_Location(grad_relu_output, location_mask)
        dL_dgamma_4, dL_dbeta_4, avg_pc_4, backward_const_4 = backward_LightNorm(grad_maxpool_output, layer4_cache)
        e = time.time()
        if DEBUG: print("Software : ",e-s)

        '''
        save_weights(ReLu_Marking_Layer4,"/home/msis/Desktop/pcie_python/GUI_list/result/ReLu_Marking_Layer4.txt")

        save_weights(grad_relu_output,"/home/msis/Desktop/pcie_python/GUI_list/result/grad_relu_output.txt")

        save_weights(grad_maxpool_output,"/home/msis/Desktop/pcie_python/GUI_list/result/grad_maxpool_output.txt")

        save_weights(Output_Grad_Layer5,"/home/msis/Desktop/pcie_python/GUI_list/result/Output_Grad_Layer5.txt")
        '''

        # avg_pc_4 = avg_pc_4.squeeze()
        # backward_const_4 = backward_const_4.squeeze()
        s = time.time()
        avg_pc_4, backward_const_4 = Mean_Var_Dec2Bfloat(avg_pc_4, backward_const_4, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat : ",e-s)

        # Weight_Backward_Layer4 for Soft2Hardware
        s = time.time()
        Weight_Backward_Layer4 = Weight_Hardware_Backward_ReOrdering_OtherLayer(256, 128, data.Weight_Bfloat[4], backward_const_4, avg_pc_4)
        e =time.time()
        if DEBUG: print("Weight Reordering : ",e-s)


        # Break 256To32 and Flip the Data: 
        s = time.time()
        Weight_Backward_Layer4_CH0 = data_256_32(Weight_Backward_Layer4[0])
        Weight_Backward_Layer4_CH1 = data_256_32(Weight_Backward_Layer4[1])
        e = time.time()
        if DEBUG: print("256bit to 32bit : ",e-s)

        # Write Weight For Backward into DDR
        s = time.time()
        Write_DDR(Weight_Backward_Layer4_CH0,Wr_Address=0x82380000)
        Write_DDR(Weight_Backward_Layer4_CH1,Wr_Address=0x92380000)
        e = time.time()
        if DEBUG: print("Write DDR : ",e-s)

        # Gradient of Beta Calculation:
        # Beta_Gradient_Layer5 = (Output_Grad_Layer5).sum(dim=(0, 2, 3), keepdim=True)

        # Weight Gradient
        s = time.time()
        Weight_Gradient1_Layer5_CH0 = Read_DDR(Rd_Address=0x8EFDC000,  End_Address=0x8F0FC000)
        Weight_Gradient1_Layer5_CH0_256 = data_32_to_16(Weight_Gradient1_Layer5_CH0)
        #if DEBUG: print("Weight_Gradient1_Layer5_CH0 : ", len(Weight_Gradient1_Layer5_CH0))   

        Weight_Gradient2_Layer5_CH0 = Read_DDR(Rd_Address=0x8F0FC000,  End_Address=0x8F21C000)
        Weight_Gradient2_Layer5_CH0_256 = data_32_to_16(Weight_Gradient2_Layer5_CH0)
        #if DEBUG: print("Weight_Gradient2_Layer5_CH0 : ", len(Weight_Gradient2_Layer5_CH0))    

        Weight_Gradient3_Layer5_CH0 = Read_DDR(Rd_Address=0x8F21C000,  End_Address=0x8F33C000)
        Weight_Gradient3_Layer5_CH0_256 = data_32_to_16(Weight_Gradient3_Layer5_CH0)
        #if DEBUG: print("Weight_Gradient3_Layer5_CH0 : ", len(Weight_Gradient3_Layer5_CH0)) 

        Weight_Gradient4_Layer5_CH0 = Read_DDR(Rd_Address=0x8F33C000,  End_Address=0x8F45C000)
        Weight_Gradient4_Layer5_CH0_256 = data_32_to_16(Weight_Gradient4_Layer5_CH0)
        #if DEBUG: print("Weight_Gradient4_Layer5_CH0 : ", len(Weight_Gradient4_Layer5_CH0)) 

        Weight_Gradient5_Layer5_CH0 = Read_DDR(Rd_Address=0x8F45C000,  End_Address=0x8F57C000)
        Weight_Gradient5_Layer5_CH0_256 = data_32_to_16(Weight_Gradient5_Layer5_CH0)
        #if DEBUG: print("Weight_Gradient5_Layer5_CH0 : ", len(Weight_Gradient5_Layer5_CH0)) 

        Weight_Gradient6_Layer5_CH0 = Read_DDR(Rd_Address=0x8F57C000,  End_Address=0x8F69C000)
        Weight_Gradient6_Layer5_CH0_256 = data_32_to_16(Weight_Gradient6_Layer5_CH0)
        #if DEBUG: print("Weight_Gradient6_Layer5_CH0 : ", len(Weight_Gradient6_Layer5_CH0)) 

        Weight_Gradient7_Layer5_CH0 = Read_DDR(Rd_Address=0x8F69C000,  End_Address=0x8F7BC000)
        Weight_Gradient7_Layer5_CH0_256 = data_32_to_16(Weight_Gradient7_Layer5_CH0)
        #if DEBUG: print("Weight_Gradient7_Layer5_CH0 : ", len(Weight_Gradient7_Layer5_CH0)) 

        Weight_Gradient8_Layer5_CH0 = Read_DDR(Rd_Address=0x8F7BC000,  End_Address=0x8F8DC000)
        Weight_Gradient8_Layer5_CH0_256 = data_32_to_16(Weight_Gradient8_Layer5_CH0)
        #if DEBUG: print("Weight_Gradient8_Layer5_CH0 : ", len(Weight_Gradient8_Layer5_CH0)) 

        Weight_Gradient1_Layer5_CH1 = Read_DDR(Rd_Address=0x9EFDC000,  End_Address=0x9F0FC000)
        Weight_Gradient1_Layer5_CH1_256 = data_32_to_16(Weight_Gradient1_Layer5_CH1)
        #if DEBUG: print("Weight_Gradient1_Layer5_CH1 : ", len(Weight_Gradient1_Layer5_CH1)) 

        Weight_Gradient2_Layer5_CH1 = Read_DDR(Rd_Address=0x9F0FC000,  End_Address=0x9F21C000)
        Weight_Gradient2_Layer5_CH1_256 = data_32_to_16(Weight_Gradient2_Layer5_CH1)
        #if DEBUG: print("Weight_Gradient2_Layer5_CH1 : ", len(Weight_Gradient2_Layer5_CH1)) 

        Weight_Gradient3_Layer5_CH1 = Read_DDR(Rd_Address=0x9F21C000,  End_Address=0x9F33C000)
        Weight_Gradient3_Layer5_CH1_256 = data_32_to_16(Weight_Gradient3_Layer5_CH1)
        #if DEBUG: print("Weight_Gradient3_Layer5_CH1 : ", len(Weight_Gradient3_Layer5_CH1)) 

        Weight_Gradient4_Layer5_CH1 = Read_DDR(Rd_Address=0x9F33C000,  End_Address=0x9F45C000)
        Weight_Gradient4_Layer5_CH1_256 = data_32_to_16(Weight_Gradient4_Layer5_CH1)
        #if DEBUG: print("Weight_Gradient4_Layer5_CH1 : ", len(Weight_Gradient4_Layer5_CH1)) 

        Weight_Gradient5_Layer5_CH1 = Read_DDR(Rd_Address=0x9F45C000,  End_Address=0x9F57C000)
        Weight_Gradient5_Layer5_CH1_256 = data_32_to_16(Weight_Gradient5_Layer5_CH1)
        #if DEBUG: print("Weight_Gradient5_Layer5_CH1 : ", len(Weight_Gradient5_Layer5_CH1)) 

        Weight_Gradient6_Layer5_CH1 = Read_DDR(Rd_Address=0x9F57C000,  End_Address=0x9F69C000)
        Weight_Gradient6_Layer5_CH1_256 = data_32_to_16(Weight_Gradient6_Layer5_CH1)
        #if DEBUG: print("Weight_Gradient6_Layer5_CH1 : ", len(Weight_Gradient6_Layer5_CH1)) 

        Weight_Gradient7_Layer5_CH1 = Read_DDR(Rd_Address=0x9F69C000,  End_Address=0x9F7BC000)
        Weight_Gradient7_Layer5_CH1_256 = data_32_to_16(Weight_Gradient7_Layer5_CH1)
        #if DEBUG: print("Weight_Gradient7_Layer5_CH1 : ", len(Weight_Gradient7_Layer5_CH1)) 

        Weight_Gradient8_Layer5_CH1 = Read_DDR(Rd_Address=0x9F7BC000,  End_Address=0x9F8DC000)
        Weight_Gradient8_Layer5_CH1_256 = data_32_to_16(Weight_Gradient8_Layer5_CH1)
        #if DEBUG: print("Weight_Gradient8_Layer5_CH1 : ", len(Weight_Gradient8_Layer5_CH1)) 
        e = time.time()
        if DEBUG: print("Read DDR & 32bit to 16bit : ",e-s)

        '''
        test_out = 'Weight_Result/Weight_Gradient1_Layer5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient1_Layer5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer5_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer5_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer5_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer5_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''

        s = time.time()
        Weight_Gradient1_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer5_CH0_256, Weight_Gradient1_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
        Weight_Gradient2_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer5_CH0_256, Weight_Gradient2_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
        Weight_Gradient3_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer5_CH0_256, Weight_Gradient3_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
        Weight_Gradient4_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer5_CH0_256, Weight_Gradient4_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
        Weight_Gradient5_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer5_CH0_256, Weight_Gradient5_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
        Weight_Gradient6_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer5_CH0_256, Weight_Gradient6_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
        Weight_Gradient7_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer5_CH0_256, Weight_Gradient7_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
        Weight_Gradient8_Layer5 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer5_CH0_256, Weight_Gradient8_Layer5_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=512, In_CH=256, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)
        
        Weight_Gradient_Layer5 = [Weight_Gradient1_Layer5, Weight_Gradient2_Layer5, Weight_Gradient3_Layer5, Weight_Gradient4_Layer5, Weight_Gradient5_Layer5, 
                                Weight_Gradient6_Layer5, Weight_Gradient7_Layer5, Weight_Gradient8_Layer5]
        
        # Weight_Gradient_Layer5 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer5)]   
        Weight_Gradient_Layer5 = list(np.mean(np.array(Weight_Gradient_Layer5), axis=0))
        
        Weight_Gradient_Layer5 = torch.tensor([float(value) for value in Weight_Gradient_Layer5], dtype=torch.float32).reshape(512, 256, 3, 3)  

        layer5_end = time.time()
        process_time = layer5_end - layer5_start
        if DEBUG: print("Layer5 Process Time : ", process_time)

        resume()
        #if DEBUG: print(irq_val)    

        #################################################
        #             Backward Layer 4 Start            #
        #################################################
        # check Layer4 IRQ
        Blayer4_start = time.time()
        check_irq_otherlayer()
        s = time.time()
        # self.app_instance .change_color(self.app_instance.L5_IRQ_canvas, self.app_instance.L5_IRQ, "green")
        # Read Gradient of Output After ReLU Backward: 
        Output_Grad1_Layer4_CH0_ = Read_DDR(Rd_Address=0x86360000,  End_Address=0x8637A000)
        Output_Grad1_Layer4_CH0_16 = data_32_to_16(Output_Grad1_Layer4_CH0_)
        #if DEBUG: print("Read Output_Grad1_Layer4_CH0")

        Output_Grad1_Layer4_CH1_ = Read_DDR(Rd_Address=0x96360000,  End_Address=0x9637A000)
        Output_Grad1_Layer4_CH1_16 = data_32_to_16(Output_Grad1_Layer4_CH1_)
        #if DEBUG: print("Read Output_Grad1_Layer4_CH1")

        Output_Grad2_Layer4_CH0_ = Read_DDR(Rd_Address=0x8637A000,  End_Address=0x86394000)
        Output_Grad2_Layer4_CH0_16 = data_32_to_16(Output_Grad2_Layer4_CH0_)
        #if DEBUG: print("Read Output_Grad2_Layer4_CH0")

        Output_Grad2_Layer4_CH1_ = Read_DDR(Rd_Address=0x9637A000,  End_Address=0x96394000)
        Output_Grad2_Layer4_CH1_16 = data_32_to_16(Output_Grad2_Layer4_CH1_)
        #if DEBUG: print("Read Output_Grad2_Layer4_CH1")

        Output_Grad3_Layer4_CH0_ = Read_DDR(Rd_Address=0x86394000,  End_Address=0x863AE000)
        Output_Grad3_Layer4_CH0_16 = data_32_to_16(Output_Grad3_Layer4_CH0_)
        #if DEBUG: print("Read Output_Grad3_Layer4_CH0")

        Output_Grad3_Layer4_CH1_ = Read_DDR(Rd_Address=0x96394000,  End_Address=0x963AE000)
        Output_Grad3_Layer4_CH1_16 = data_32_to_16(Output_Grad3_Layer4_CH1_)
        #if DEBUG: print("Read Output_Grad3_Layer4_CH1")

        Output_Grad4_Layer4_CH0_ = Read_DDR(Rd_Address=0x863AE000,  End_Address=0x863C8000)
        Output_Grad4_Layer4_CH0_16 = data_32_to_16(Output_Grad4_Layer4_CH0_)
        #if DEBUG: print("Read Output_Grad4_Layer4_CH0")

        Output_Grad4_Layer4_CH1_ = Read_DDR(Rd_Address=0x963AE000,  End_Address=0x963C8000)
        Output_Grad4_Layer4_CH1_16 = data_32_to_16(Output_Grad4_Layer4_CH1_)
        #if DEBUG: print("Read Output_Grad4_Layer4_CH1")

        Output_Grad5_Layer4_CH0_ = Read_DDR(Rd_Address=0x863C8000,  End_Address=0x863E2000)
        Output_Grad5_Layer4_CH0_16 = data_32_to_16(Output_Grad5_Layer4_CH0_)
        #if DEBUG: print("Read Output_Grad5_Layer4_CH0")

        Output_Grad5_Layer4_CH1_ = Read_DDR(Rd_Address=0x963C8000,  End_Address=0x963E2000)
        Output_Grad5_Layer4_CH1_16 = data_32_to_16(Output_Grad5_Layer4_CH1_)
        #if DEBUG: print("Read Output_Grad5_Layer4_CH1")

        Output_Grad6_Layer4_CH0_ = Read_DDR(Rd_Address=0x863E2000,  End_Address=0x863FC000)
        Output_Grad6_Layer4_CH0_16 = data_32_to_16(Output_Grad6_Layer4_CH0_)
        #if DEBUG: print("Read Output_Grad6_Layer4_CH0")

        Output_Grad6_Layer4_CH1_ = Read_DDR(Rd_Address=0x963E2000,  End_Address=0x963FC000)
        Output_Grad6_Layer4_CH1_16 = data_32_to_16(Output_Grad6_Layer4_CH1_)
        #if DEBUG: print("Read Output_Grad6_Layer4_CH1")

        Output_Grad7_Layer4_CH0_ = Read_DDR(Rd_Address=0x863FC000,  End_Address=0x86416000)
        Output_Grad7_Layer4_CH0_16 = data_32_to_16(Output_Grad7_Layer4_CH0_)
        #if DEBUG: print("Read Output_Grad7_Layer4_CH0")

        Output_Grad7_Layer4_CH1_ = Read_DDR(Rd_Address=0x963FC000,  End_Address=0x96416000)
        Output_Grad7_Layer4_CH1_16 = data_32_to_16(Output_Grad7_Layer4_CH1_)
        #if DEBUG: print("Read Output_Grad7_Layer4_CH1")

        Output_Grad8_Layer4_CH0_ = Read_DDR(Rd_Address=0x86416000,  End_Address=0x86430000)
        Output_Grad8_Layer4_CH0_16 = data_32_to_16(Output_Grad8_Layer4_CH0_)
        #if DEBUG: print("Read Output_Grad8_Layer4_CH0")

        Output_Grad8_Layer4_CH1_ = Read_DDR(Rd_Address=0x96416000,  End_Address=0x96430000)
        Output_Grad8_Layer4_CH1_16 = data_32_to_16(Output_Grad8_Layer4_CH1_)
        #if DEBUG: print("Read Output_Grad8_Layer4_CH1")
        e = time.time()
        if DEBUG: print("Read OG DDR & 32bit to 16bit : ",e-s)

        s = time.time()
        Output_Grad1_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer4_CH0_16, Output_Grad1_Layer4_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
        Output_Grad2_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer4_CH0_16, Output_Grad2_Layer4_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
        Output_Grad3_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer4_CH0_16, Output_Grad3_Layer4_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
        Output_Grad4_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer4_CH0_16, Output_Grad4_Layer4_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
        Output_Grad5_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer4_CH0_16, Output_Grad5_Layer4_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
        Output_Grad6_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer4_CH0_16, Output_Grad6_Layer4_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
        Output_Grad7_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer4_CH0_16, Output_Grad7_Layer4_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
        Output_Grad8_Layer4 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer4_CH0_16, Output_Grad8_Layer4_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
        e = time.time()
        if DEBUG: print("Bflaot to Dec : ",e-s)
        
        Output_Grads_Layer4 = Output_Grad1_Layer4 + Output_Grad2_Layer4 + Output_Grad3_Layer4 + Output_Grad4_Layer4 + \
                                Output_Grad5_Layer4 + Output_Grad6_Layer4 + Output_Grad7_Layer4 + Output_Grad8_Layer4    
        Output_Grad_Layer4 = torch.tensor([float(value) for value in Output_Grads_Layer4], dtype=torch.float32).reshape(8, 128, 26, 26)

        # BReLu Marking
        s = time.time()
        ReLu_Marking1_Layer3_CH0 = Read_DDR(Rd_Address=0x87D2C000,  End_Address=0x87D46000)
        ReLu_Marking1_Layer3_CH0_256 = data_32_to_16(ReLu_Marking1_Layer3_CH0)

        ReLu_Marking2_Layer3_CH0 = Read_DDR(Rd_Address=0x87D46000,  End_Address=0x87D60000)
        ReLu_Marking2_Layer3_CH0_256 = data_32_to_16(ReLu_Marking2_Layer3_CH0)

        ReLu_Marking3_Layer3_CH0 = Read_DDR(Rd_Address=0x87D60000,  End_Address=0x87D7A000)
        ReLu_Marking3_Layer3_CH0_256 = data_32_to_16(ReLu_Marking3_Layer3_CH0)

        ReLu_Marking4_Layer3_CH0 = Read_DDR(Rd_Address=0x87D7A000,  End_Address=0x87D94000)
        ReLu_Marking4_Layer3_CH0_256 = data_32_to_16(ReLu_Marking4_Layer3_CH0)

        ReLu_Marking5_Layer3_CH0 = Read_DDR(Rd_Address=0x87D94000,  End_Address=0x87DAE000)
        ReLu_Marking5_Layer3_CH0_256 = data_32_to_16(ReLu_Marking5_Layer3_CH0)

        ReLu_Marking6_Layer3_CH0 = Read_DDR(Rd_Address=0x87DAE000,  End_Address=0x87DC8000)
        ReLu_Marking6_Layer3_CH0_256 = data_32_to_16(ReLu_Marking6_Layer3_CH0)

        ReLu_Marking7_Layer3_CH0 = Read_DDR(Rd_Address=0x87DC8000,  End_Address=0x87DE2000)
        ReLu_Marking7_Layer3_CH0_256 = data_32_to_16(ReLu_Marking7_Layer3_CH0)

        ReLu_Marking8_Layer3_CH0 = Read_DDR(Rd_Address=0x87DE2000,  End_Address=0x87DFC000)
        ReLu_Marking8_Layer3_CH0_256 = data_32_to_16(ReLu_Marking8_Layer3_CH0)

        ReLu_Marking1_Layer3_CH1 = Read_DDR(Rd_Address=0x97D2C000,  End_Address=0x97D46000)
        ReLu_Marking1_Layer3_CH1_256 = data_32_to_16(ReLu_Marking1_Layer3_CH1)

        ReLu_Marking2_Layer3_CH1 = Read_DDR(Rd_Address=0x97D46000,  End_Address=0x97D60000)
        ReLu_Marking2_Layer3_CH1_256 = data_32_to_16(ReLu_Marking2_Layer3_CH1)

        ReLu_Marking3_Layer3_CH1 = Read_DDR(Rd_Address=0x97D60000,  End_Address=0x97D7A000)
        ReLu_Marking3_Layer3_CH1_256 = data_32_to_16(ReLu_Marking3_Layer3_CH1)

        ReLu_Marking4_Layer3_CH1 = Read_DDR(Rd_Address=0x97D7A000,  End_Address=0x97D94000)
        ReLu_Marking4_Layer3_CH1_256 = data_32_to_16(ReLu_Marking4_Layer3_CH1)

        ReLu_Marking5_Layer3_CH1 = Read_DDR(Rd_Address=0x97D94000,  End_Address=0x97DAE000)
        ReLu_Marking5_Layer3_CH1_256 = data_32_to_16(ReLu_Marking5_Layer3_CH1)

        ReLu_Marking6_Layer3_CH1 = Read_DDR(Rd_Address=0x97DAE000,  End_Address=0x97DC8000)
        ReLu_Marking6_Layer3_CH1_256 = data_32_to_16(ReLu_Marking6_Layer3_CH1)

        ReLu_Marking7_Layer3_CH1 = Read_DDR(Rd_Address=0x97DC8000,  End_Address=0x97DE2000)
        ReLu_Marking7_Layer3_CH1_256 = data_32_to_16(ReLu_Marking7_Layer3_CH1)

        ReLu_Marking8_Layer3_CH1 = Read_DDR(Rd_Address=0x97DE2000,  End_Address=0x97DFC000)
        ReLu_Marking8_Layer3_CH1_256 = data_32_to_16(ReLu_Marking8_Layer3_CH1)
        e = time.time()
        if DEBUG: print("Read RM DDR & 32bit to 16bit : ",e-s)

        # ReLu Reordering
        s = time.time()
        ReLu_Marking1_Layer3 = Read_ReLu_Marking(ReLu_Marking1_Layer3_CH0_256, ReLu_Marking1_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
        ReLu_Marking2_Layer3 = Read_ReLu_Marking(ReLu_Marking2_Layer3_CH0_256, ReLu_Marking2_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
        ReLu_Marking3_Layer3 = Read_ReLu_Marking(ReLu_Marking3_Layer3_CH0_256, ReLu_Marking3_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
        ReLu_Marking4_Layer3 = Read_ReLu_Marking(ReLu_Marking4_Layer3_CH0_256, ReLu_Marking4_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
        ReLu_Marking5_Layer3 = Read_ReLu_Marking(ReLu_Marking5_Layer3_CH0_256, ReLu_Marking5_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
        ReLu_Marking6_Layer3 = Read_ReLu_Marking(ReLu_Marking6_Layer3_CH0_256, ReLu_Marking6_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
        ReLu_Marking7_Layer3 = Read_ReLu_Marking(ReLu_Marking7_Layer3_CH0_256, ReLu_Marking7_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
        ReLu_Marking8_Layer3 = Read_ReLu_Marking(ReLu_Marking8_Layer3_CH0_256, ReLu_Marking8_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, Out_Size=26, Layer8=False)
        e = time.time()
        if DEBUG: print("ReLu Reordering : ",e-s)

        ReLu_Marking_Layer3 = ReLu_Marking1_Layer3 + ReLu_Marking2_Layer3 + ReLu_Marking3_Layer3 + ReLu_Marking4_Layer3 + ReLu_Marking5_Layer3 + \
                                ReLu_Marking6_Layer3 + ReLu_Marking7_Layer3 + ReLu_Marking8_Layer3
        
        ReLu_Marking_Layer3 = torch.tensor([float(value) for value in ReLu_Marking_Layer3], dtype=torch.float32).reshape(8, 128, 26, 26)

        # BReLu Calculate
        # Output_Grad_layer4_input = torch.tensor(Output_Grad_Layer4, dtype=torch.float32).reshape(8,128,26,26)
        # Layer3_Location = torch.tensor(ReLu_Marking_Layer3, dtype=torch.float32).reshape(8,128,26,26)

        s = time.time()
        relu_mask, location_mask = split_location(ReLu_Marking_Layer3)
        grad_relu_output = backward_active(Output_Grad_Layer4, relu_mask)
        grad_maxpool_output = backward_MaxPool_Location(grad_relu_output, location_mask)
        dL_dgamma_3, dL_dbeta_3, avg_pc_3, backward_const_3 = backward_LightNorm(grad_maxpool_output, layer3_cache)
        e = time.time()
        if DEBUG: print("Software : ",e-s)

        # avg_pc_3 = avg_pc_3.squeeze()
        # backward_const_3 = backward_const_3.squeeze()
        s = time.time()
        avg_pc_3, backward_const_3 = Mean_Var_Dec2Bfloat(avg_pc_3, backward_const_3, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat : ",e-s)

        # Weight_Backward_Layer3 for Soft2Hardware
        s = time.time()
        Weight_Backward_Layer3 = Weight_Hardware_Backward_ReOrdering_OtherLayer(128, 64, data.Weight_Bfloat[3], backward_const_3, avg_pc_3)
        e = time.time()
        if DEBUG: print("Weight Reordering : ",e-s)


        # Break 256To32 and Flip the Data: 
        s = time.time()
        Weight_Backward_Layer3_CH0 = data_256_32(Weight_Backward_Layer3[0])
        Weight_Backward_Layer3_CH1 = data_256_32(Weight_Backward_Layer3[1])
        e = time.time()
        if DEBUG: print("256bit to 32bit : ",e-s)

        # Write Weight For Backward into DDR
        s = time.time()
        Write_DDR(Weight_Backward_Layer3_CH0,Wr_Address=0x823D0000)
        Write_DDR(Weight_Backward_Layer3_CH1,Wr_Address=0x923D0000)
        e = time.time()
        if DEBUG: print("Write DDR : ",e-s)


        # Gradient of Beta Calculation:
        # Beta_Gradient_Layer4 = (Output_Grad_Layer4).sum(dim=(0, 2, 3), keepdim=True)

        # Weight Gradient
        s = time.time()
        Weight_Gradient1_Layer4_CH0 = Read_DDR(Rd_Address=0x8F8DC000,  End_Address=0x8F924000)
        Weight_Gradient1_Layer4_CH0_256 = data_32_to_16(Weight_Gradient1_Layer4_CH0)
        #if DEBUG: print("Weight_Gradient1_Layer4_CH0 : ", len(Weight_Gradient1_Layer4_CH0))   

        Weight_Gradient2_Layer4_CH0 = Read_DDR(Rd_Address=0x8F924000,  End_Address=0x8F96C000)
        Weight_Gradient2_Layer4_CH0_256 = data_32_to_16(Weight_Gradient2_Layer4_CH0)
        #if DEBUG: print("Weight_Gradient2_Layer4_CH0 : ", len(Weight_Gradient2_Layer4_CH0))    

        Weight_Gradient3_Layer4_CH0 = Read_DDR(Rd_Address=0x8F96C000,  End_Address=0x8F9B4000)
        Weight_Gradient3_Layer4_CH0_256 = data_32_to_16(Weight_Gradient3_Layer4_CH0)
        #if DEBUG: print("Weight_Gradient3_Layer4_CH0 : ", len(Weight_Gradient3_Layer4_CH0)) 

        Weight_Gradient4_Layer4_CH0 = Read_DDR(Rd_Address=0x8F9B4000,  End_Address=0x8F9FC000)
        Weight_Gradient4_Layer4_CH0_256 = data_32_to_16(Weight_Gradient4_Layer4_CH0)
        #if DEBUG: print("Weight_Gradient4_Layer4_CH0 : ", len(Weight_Gradient4_Layer4_CH0)) 

        Weight_Gradient5_Layer4_CH0 = Read_DDR(Rd_Address=0x8F9FC000,  End_Address=0x8FA44000)
        Weight_Gradient5_Layer4_CH0_256 = data_32_to_16(Weight_Gradient5_Layer4_CH0)
        #if DEBUG: print("Weight_Gradient5_Layer4_CH0 : ", len(Weight_Gradient5_Layer4_CH0)) 

        Weight_Gradient6_Layer4_CH0 = Read_DDR(Rd_Address=0x8FA44000,  End_Address=0x8FA8C000)
        Weight_Gradient6_Layer4_CH0_256 = data_32_to_16(Weight_Gradient6_Layer4_CH0)
        #if DEBUG: print("Weight_Gradient6_Layer4_CH0 : ", len(Weight_Gradient6_Layer4_CH0)) 

        Weight_Gradient7_Layer4_CH0 = Read_DDR(Rd_Address=0x8FA8C000,  End_Address=0x8FAD4000)
        Weight_Gradient7_Layer4_CH0_256 = data_32_to_16(Weight_Gradient7_Layer4_CH0)
        #if DEBUG: print("Weight_Gradient7_Layer4_CH0 : ", len(Weight_Gradient7_Layer4_CH0)) 

        Weight_Gradient8_Layer4_CH0 = Read_DDR(Rd_Address=0x8FAD4000,  End_Address=0x8FB1C000)
        Weight_Gradient8_Layer4_CH0_256 = data_32_to_16(Weight_Gradient8_Layer4_CH0)
        #if DEBUG: print("Weight_Gradient8_Layer4_CH0 : ", len(Weight_Gradient8_Layer4_CH0)) 

        Weight_Gradient1_Layer4_CH1 = Read_DDR(Rd_Address=0x9F8DC000,  End_Address=0x9F924000)
        Weight_Gradient1_Layer4_CH1_256 = data_32_to_16(Weight_Gradient1_Layer4_CH1)
        #if DEBUG: print("Weight_Gradient1_Layer4_CH1 : ", len(Weight_Gradient1_Layer4_CH1)) 

        Weight_Gradient2_Layer4_CH1 = Read_DDR(Rd_Address=0x9F924000,  End_Address=0x9F96C000)
        Weight_Gradient2_Layer4_CH1_256 = data_32_to_16(Weight_Gradient2_Layer4_CH1)
        #if DEBUG: print("Weight_Gradient2_Layer4_CH1 : ", len(Weight_Gradient2_Layer4_CH1)) 

        Weight_Gradient3_Layer4_CH1 = Read_DDR(Rd_Address=0x9F96C000,  End_Address=0x9F9B4000)
        Weight_Gradient3_Layer4_CH1_256 = data_32_to_16(Weight_Gradient3_Layer4_CH1)
        #if DEBUG: print("Weight_Gradient3_Layer4_CH1 : ", len(Weight_Gradient3_Layer4_CH1)) 

        Weight_Gradient4_Layer4_CH1 = Read_DDR(Rd_Address=0x9F9B4000,  End_Address=0x9F9FC000)
        Weight_Gradient4_Layer4_CH1_256 = data_32_to_16(Weight_Gradient4_Layer4_CH1)
        #if DEBUG: print("Weight_Gradient4_Layer4_CH1 : ", len(Weight_Gradient4_Layer4_CH1)) 

        Weight_Gradient5_Layer4_CH1 = Read_DDR(Rd_Address=0x9F9FC000,  End_Address=0x9FA44000)
        Weight_Gradient5_Layer4_CH1_256 = data_32_to_16(Weight_Gradient5_Layer4_CH1)
        #if DEBUG: print("Weight_Gradient5_Layer4_CH1 : ", len(Weight_Gradient5_Layer4_CH1)) 

        Weight_Gradient6_Layer4_CH1 = Read_DDR(Rd_Address=0x9FA44000,  End_Address=0x9FA8C000)
        Weight_Gradient6_Layer4_CH1_256 = data_32_to_16(Weight_Gradient6_Layer4_CH1)
        #if DEBUG: print("Weight_Gradient6_Layer4_CH1 : ", len(Weight_Gradient6_Layer4_CH1)) 

        Weight_Gradient7_Layer4_CH1 = Read_DDR(Rd_Address=0x9FA8C000,  End_Address=0x9FAD4000)
        Weight_Gradient7_Layer4_CH1_256 = data_32_to_16(Weight_Gradient7_Layer4_CH1)
        #if DEBUG: print("Weight_Gradient7_Layer4_CH1 : ", len(Weight_Gradient7_Layer4_CH1)) 

        Weight_Gradient8_Layer4_CH1 = Read_DDR(Rd_Address=0x9FAD4000,  End_Address=0x9FB1C000)
        Weight_Gradient8_Layer4_CH1_256 = data_32_to_16(Weight_Gradient8_Layer4_CH1)
        #if DEBUG: print("Weight_Gradient8_Layer4_CH1 : ", len(Weight_Gradient8_Layer4_CH1)) 
        e = time.time()
        if DEBUG: print("Read WG DDR & 32bit to 16bit : ",e-s)

        '''
        test_out = 'Weight_Result/Weight_Gradient1_Layer4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient1_Layer4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer4_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer4_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer4_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer4_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''

        s = time.time()
        Weight_Gradient1_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer4_CH0_256, Weight_Gradient1_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
        Weight_Gradient2_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer4_CH0_256, Weight_Gradient2_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
        Weight_Gradient3_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer4_CH0_256, Weight_Gradient3_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
        Weight_Gradient4_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer4_CH0_256, Weight_Gradient4_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
        Weight_Gradient5_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer4_CH0_256, Weight_Gradient5_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
        Weight_Gradient6_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer4_CH0_256, Weight_Gradient6_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
        Weight_Gradient7_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer4_CH0_256, Weight_Gradient7_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
        Weight_Gradient8_Layer4 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer4_CH0_256, Weight_Gradient8_Layer4_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=256, In_CH=128, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)

        Weight_Gradient_Layer4 = [Weight_Gradient1_Layer4, Weight_Gradient2_Layer4, Weight_Gradient3_Layer4, Weight_Gradient4_Layer4, Weight_Gradient5_Layer4, 
                                Weight_Gradient6_Layer4, Weight_Gradient7_Layer4, Weight_Gradient8_Layer4]
        
        # Weight_Gradient_Layer4 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer4)]   
        Weight_Gradient_Layer4 = list(np.mean(np.array(Weight_Gradient_Layer4), axis=0))
        
        Weight_Gradient_Layer4 = torch.tensor([float(value) for value in Weight_Gradient_Layer4], dtype=torch.float32).reshape(256, 128, 3, 3)   

        Blayer4_end = time.time()
        if DEBUG: print("Layer4 Process Time : ",Blayer4_end-Blayer4_start)

        resume()

        #################################################
        #             Backward Layer 3 Start            #
        #################################################
        # check Layer3 IRQ
        Blayer3_start = time.time()
        check_irq_otherlayer()
        s = time.time()
        # self.app_instance .change_color(self.app_instance.L4_IRQ_canvas, self.app_instance.L4_IRQ, "green")
        # Read Gradient of Output After ReLU Backward: 
        Output_Grad1_Layer3_CH0_ = Read_DDR(Rd_Address=0x85B40000,  End_Address=0x85B74000)
        Output_Grad1_Layer3_CH0_16 = data_32_to_16(Output_Grad1_Layer3_CH0_)

        Output_Grad1_Layer3_CH1_ = Read_DDR(Rd_Address=0x95B40000,  End_Address=0x95B74000)
        Output_Grad1_Layer3_CH1_16 = data_32_to_16(Output_Grad1_Layer3_CH1_)

        Output_Grad2_Layer3_CH0_ = Read_DDR(Rd_Address=0x85B74000,  End_Address=0x85BA8000)
        Output_Grad2_Layer3_CH0_16 = data_32_to_16(Output_Grad2_Layer3_CH0_)

        Output_Grad2_Layer3_CH1_ = Read_DDR(Rd_Address=0x95B74000,  End_Address=0x95BA8000)
        Output_Grad2_Layer3_CH1_16 = data_32_to_16(Output_Grad2_Layer3_CH1_)

        Output_Grad3_Layer3_CH0_ = Read_DDR(Rd_Address=0x85BA8000,  End_Address=0x85BDC000)
        Output_Grad3_Layer3_CH0_16 = data_32_to_16(Output_Grad3_Layer3_CH0_)

        Output_Grad3_Layer3_CH1_ = Read_DDR(Rd_Address=0x95BA8000,  End_Address=0x95BDC000)
        Output_Grad3_Layer3_CH1_16 = data_32_to_16(Output_Grad3_Layer3_CH1_)

        Output_Grad4_Layer3_CH0_ = Read_DDR(Rd_Address=0x85BDC000,  End_Address=0x85C10000)
        Output_Grad4_Layer3_CH0_16 = data_32_to_16(Output_Grad4_Layer3_CH0_)

        Output_Grad4_Layer3_CH1_ = Read_DDR(Rd_Address=0x95BDC000,  End_Address=0x95C10000)
        Output_Grad4_Layer3_CH1_16 = data_32_to_16(Output_Grad4_Layer3_CH1_)

        Output_Grad5_Layer3_CH0_ = Read_DDR(Rd_Address=0x85C10000,  End_Address=0x85C44000)
        Output_Grad5_Layer3_CH0_16 = data_32_to_16(Output_Grad5_Layer3_CH0_)

        Output_Grad5_Layer3_CH1_ = Read_DDR(Rd_Address=0x95C10000,  End_Address=0x95C44000)
        Output_Grad5_Layer3_CH1_16 = data_32_to_16(Output_Grad5_Layer3_CH1_)

        Output_Grad6_Layer3_CH0_ = Read_DDR(Rd_Address=0x85C44000,  End_Address=0x85C78000)
        Output_Grad6_Layer3_CH0_16 = data_32_to_16(Output_Grad6_Layer3_CH0_)

        Output_Grad6_Layer3_CH1_ = Read_DDR(Rd_Address=0x95C44000,  End_Address=0x95C78000)
        Output_Grad6_Layer3_CH1_16 = data_32_to_16(Output_Grad6_Layer3_CH1_)

        Output_Grad7_Layer3_CH0_ = Read_DDR(Rd_Address=0x85C78000,  End_Address=0x85CAC000)
        Output_Grad7_Layer3_CH0_16 = data_32_to_16(Output_Grad7_Layer3_CH0_)

        Output_Grad7_Layer3_CH1_ = Read_DDR(Rd_Address=0x95C78000,  End_Address=0x95CAC000)
        Output_Grad7_Layer3_CH1_16 = data_32_to_16(Output_Grad7_Layer3_CH1_)

        Output_Grad8_Layer3_CH0_ = Read_DDR(Rd_Address=0x85CAC000,  End_Address=0x85CE0000)
        Output_Grad8_Layer3_CH0_16 = data_32_to_16(Output_Grad8_Layer3_CH0_)

        Output_Grad8_Layer3_CH1_ = Read_DDR(Rd_Address=0x95CAC000,  End_Address=0x95CE0000)
        Output_Grad8_Layer3_CH1_16 = data_32_to_16(Output_Grad8_Layer3_CH1_)
        e = time.time()
        if DEBUG: print("Read OG DDR & 32bit to 16bit : ",e-s)

        s = time.time()
        Output_Grad1_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer3_CH0_16, Output_Grad1_Layer3_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
        Output_Grad2_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer3_CH0_16, Output_Grad2_Layer3_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
        Output_Grad3_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer3_CH0_16, Output_Grad3_Layer3_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
        Output_Grad4_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer3_CH0_16, Output_Grad4_Layer3_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
        Output_Grad5_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer3_CH0_16, Output_Grad5_Layer3_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
        Output_Grad6_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer3_CH0_16, Output_Grad6_Layer3_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
        Output_Grad7_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer3_CH0_16, Output_Grad7_Layer3_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
        Output_Grad8_Layer3 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer3_CH0_16, Output_Grad8_Layer3_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
        e = time.time()
        if DEBUG: print("Bflaot to Dec : ",e-s)
        
        Output_Grads_Layer3 = Output_Grad1_Layer3 + Output_Grad2_Layer3 + Output_Grad3_Layer3 + Output_Grad4_Layer3 + \
                                Output_Grad5_Layer3 + Output_Grad6_Layer3 + Output_Grad7_Layer3 + Output_Grad8_Layer3    
        Output_Grad_Layer3 = torch.tensor([float(value) for value in Output_Grads_Layer3], dtype=torch.float32).reshape(8, 64, 52, 52)

        # BReLu Marking
        s = time.time()
        ReLu_Marking1_Layer2_CH0 = Read_DDR(Rd_Address=0x87B8C000,  End_Address=0x87BC0000)
        ReLu_Marking1_Layer2_CH0_256 = data_32_to_16(ReLu_Marking1_Layer2_CH0)

        ReLu_Marking2_Layer2_CH0 = Read_DDR(Rd_Address=0x87BC0000,  End_Address=0x87BF4000)
        ReLu_Marking2_Layer2_CH0_256 = data_32_to_16(ReLu_Marking2_Layer2_CH0)

        ReLu_Marking3_Layer2_CH0 = Read_DDR(Rd_Address=0x87BF4000,  End_Address=0x87C28000)
        ReLu_Marking3_Layer2_CH0_256 = data_32_to_16(ReLu_Marking3_Layer2_CH0)

        ReLu_Marking4_Layer2_CH0 = Read_DDR(Rd_Address=0x87C28000,  End_Address=0x87C5C000)
        ReLu_Marking4_Layer2_CH0_256 = data_32_to_16(ReLu_Marking4_Layer2_CH0)

        ReLu_Marking5_Layer2_CH0 = Read_DDR(Rd_Address=0x87C5C000,  End_Address=0x87C90000)
        ReLu_Marking5_Layer2_CH0_256 = data_32_to_16(ReLu_Marking5_Layer2_CH0)

        ReLu_Marking6_Layer2_CH0 = Read_DDR(Rd_Address=0x87C90000,  End_Address=0x87CC4000)
        ReLu_Marking6_Layer2_CH0_256 = data_32_to_16(ReLu_Marking6_Layer2_CH0)

        ReLu_Marking7_Layer2_CH0 = Read_DDR(Rd_Address=0x87CC4000,  End_Address=0x87CF8000)
        ReLu_Marking7_Layer2_CH0_256 = data_32_to_16(ReLu_Marking7_Layer2_CH0)

        ReLu_Marking8_Layer2_CH0 = Read_DDR(Rd_Address=0x87CF8000,  End_Address=0x87D2C000)
        ReLu_Marking8_Layer2_CH0_256 = data_32_to_16(ReLu_Marking8_Layer2_CH0)

        ReLu_Marking1_Layer2_CH1 = Read_DDR(Rd_Address=0x97B8C000,  End_Address=0x97BC0000)
        ReLu_Marking1_Layer2_CH1_256 = data_32_to_16(ReLu_Marking1_Layer2_CH1)

        ReLu_Marking2_Layer2_CH1 = Read_DDR(Rd_Address=0x97BC0000,  End_Address=0x97BF4000)
        ReLu_Marking2_Layer2_CH1_256 = data_32_to_16(ReLu_Marking2_Layer2_CH1)

        ReLu_Marking3_Layer2_CH1 = Read_DDR(Rd_Address=0x97BF4000,  End_Address=0x97C28000)
        ReLu_Marking3_Layer2_CH1_256 = data_32_to_16(ReLu_Marking3_Layer2_CH1)

        ReLu_Marking4_Layer2_CH1 = Read_DDR(Rd_Address=0x97C28000,  End_Address=0x97C5C000)
        ReLu_Marking4_Layer2_CH1_256 = data_32_to_16(ReLu_Marking4_Layer2_CH1)

        ReLu_Marking5_Layer2_CH1 = Read_DDR(Rd_Address=0x97C5C000,  End_Address=0x97C90000)
        ReLu_Marking5_Layer2_CH1_256 = data_32_to_16(ReLu_Marking5_Layer2_CH1)

        ReLu_Marking6_Layer2_CH1 = Read_DDR(Rd_Address=0x97C90000,  End_Address=0x97CC4000)
        ReLu_Marking6_Layer2_CH1_256 = data_32_to_16(ReLu_Marking6_Layer2_CH1)

        ReLu_Marking7_Layer2_CH1 = Read_DDR(Rd_Address=0x97CC4000,  End_Address=0x97CF8000)
        ReLu_Marking7_Layer2_CH1_256 = data_32_to_16(ReLu_Marking7_Layer2_CH1)

        ReLu_Marking8_Layer2_CH1 = Read_DDR(Rd_Address=0x97CF8000,  End_Address=0x97D2C000)
        ReLu_Marking8_Layer2_CH1_256 = data_32_to_16(ReLu_Marking8_Layer2_CH1)
        e = time.time()
        if DEBUG: print("Read RM DDR & 32bit to 16bit : ",e-s)

        # ReLu Reordering
        s = time.time()
        ReLu_Marking1_Layer2 = Read_ReLu_Marking(ReLu_Marking1_Layer2_CH0_256, ReLu_Marking1_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
        ReLu_Marking2_Layer2 = Read_ReLu_Marking(ReLu_Marking2_Layer2_CH0_256, ReLu_Marking2_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
        ReLu_Marking3_Layer2 = Read_ReLu_Marking(ReLu_Marking3_Layer2_CH0_256, ReLu_Marking3_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
        ReLu_Marking4_Layer2 = Read_ReLu_Marking(ReLu_Marking4_Layer2_CH0_256, ReLu_Marking4_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
        ReLu_Marking5_Layer2 = Read_ReLu_Marking(ReLu_Marking5_Layer2_CH0_256, ReLu_Marking5_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
        ReLu_Marking6_Layer2 = Read_ReLu_Marking(ReLu_Marking6_Layer2_CH0_256, ReLu_Marking6_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
        ReLu_Marking7_Layer2 = Read_ReLu_Marking(ReLu_Marking7_Layer2_CH0_256, ReLu_Marking7_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
        ReLu_Marking8_Layer2 = Read_ReLu_Marking(ReLu_Marking8_Layer2_CH0_256, ReLu_Marking8_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, Out_Size=52, Layer8=False)
        e = time.time()
        if DEBUG: print("ReLu Reordering : ",e-s)

        ReLu_Marking_Layer2 = ReLu_Marking1_Layer2 + ReLu_Marking2_Layer2 + ReLu_Marking3_Layer2 + ReLu_Marking4_Layer2 + ReLu_Marking5_Layer2 + \
                                ReLu_Marking6_Layer2 + ReLu_Marking7_Layer2 + ReLu_Marking8_Layer2
        
        ReLu_Marking_Layer2 = torch.tensor([float(value) for value in ReLu_Marking_Layer2], dtype=torch.float32).reshape(8, 64, 52, 52)

        # BReLu Calculate
        # Output_Grad_layer3_input = torch.tensor(Output_Grad_Layer3, dtype=torch.float32).reshape(8,64,52,52)
        # Layer2_Location = torch.tensor(ReLu_Marking_Layer2, dtype=torch.float32).reshape(8,64,52,52)

        s = time.time()
        relu_mask, location_mask = split_location(ReLu_Marking_Layer2)
        grad_relu_output = backward_active(Output_Grad_Layer3, relu_mask)
        grad_maxpool_output = backward_MaxPool_Location(grad_relu_output, location_mask)
        dL_dgamma_2, dL_dbeta_2, avg_pc_2, backward_const_2 = backward_LightNorm(grad_maxpool_output, layer2_cache)
        e = time.time()
        if DEBUG: print("Software : ",e-s)

        # avg_pc_2 = avg_pc_2.squeeze()
        # backward_const_2 = backward_const_2.squeeze()
        s = time.time()
        avg_pc_2, backward_const_2 = Mean_Var_Dec2Bfloat(avg_pc_2, backward_const_2, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat : ",e-s)

        # Weight_Backward_Layer2 for Soft2Hardware
        s = time.time()
        Weight_Backward_Layer2 = Weight_Hardware_Backward_ReOrdering_OtherLayer(64, 32, data.Weight_Bfloat[2], backward_const_2, avg_pc_2)
        e = time.time()
        if DEBUG: print("Weight Reordering : ",e-s)

        # Break 256To32 and Flip the Data: 
        s = time.time()
        Weight_Backward_Layer2_CH0 = data_256_32(Weight_Backward_Layer2[0])
        Weight_Backward_Layer2_CH1 = data_256_32(Weight_Backward_Layer2[1])
        e = time.time()
        if DEBUG: print("256bit to 32bit : ",e-s)

        # Write Weight For Backward into DDR
        s = time.time()
        Write_DDR(Weight_Backward_Layer2_CH0,Wr_Address=0x823E4000)
        Write_DDR(Weight_Backward_Layer2_CH1,Wr_Address=0x923E4000)
        e = time.time()
        if DEBUG: print("Write DDR : ",e-s)


        # Gradient of Beta Calculation:
        # Beta_Gradient_Layer3 = (Output_Grad_Layer3).sum(dim=(0, 2, 3), keepdim=True)

        # Weight Gradient
        s = time.time()
        Weight_Gradient1_Layer3_CH0 = Read_DDR(Rd_Address=0x8FB1C000,  End_Address=0x8FB2E000)
        Weight_Gradient1_Layer3_CH0_256 = data_32_to_16(Weight_Gradient1_Layer3_CH0)
        #if DEBUG: print("Weight_Gradient1_Layer3_CH0 : ", len(Weight_Gradient1_Layer3_CH0))   

        Weight_Gradient2_Layer3_CH0 = Read_DDR(Rd_Address=0x8FB2E000,  End_Address=0x8FB40000)
        Weight_Gradient2_Layer3_CH0_256 = data_32_to_16(Weight_Gradient2_Layer3_CH0)
        #if DEBUG: print("Weight_Gradient2_Layer3_CH0 : ", len(Weight_Gradient2_Layer3_CH0))    

        Weight_Gradient3_Layer3_CH0 = Read_DDR(Rd_Address=0x8FB40000,  End_Address=0x8FB52000)
        Weight_Gradient3_Layer3_CH0_256 = data_32_to_16(Weight_Gradient3_Layer3_CH0)
        #if DEBUG: print("Weight_Gradient3_Layer3_CH0 : ", len(Weight_Gradient3_Layer3_CH0)) 

        Weight_Gradient4_Layer3_CH0 = Read_DDR(Rd_Address=0x8FB52000,  End_Address=0x8FB64000)
        Weight_Gradient4_Layer3_CH0_256 = data_32_to_16(Weight_Gradient4_Layer3_CH0)
        #if DEBUG: print("Weight_Gradient4_Layer3_CH0 : ", len(Weight_Gradient4_Layer3_CH0)) 

        Weight_Gradient5_Layer3_CH0 = Read_DDR(Rd_Address=0x8FB64000,  End_Address=0x8FB76000)
        Weight_Gradient5_Layer3_CH0_256 = data_32_to_16(Weight_Gradient5_Layer3_CH0)
        #if DEBUG: print("Weight_Gradient5_Layer3_CH0 : ", len(Weight_Gradient5_Layer3_CH0)) 

        Weight_Gradient6_Layer3_CH0 = Read_DDR(Rd_Address=0x8FB76000,  End_Address=0x8FB88000)
        Weight_Gradient6_Layer3_CH0_256 = data_32_to_16(Weight_Gradient6_Layer3_CH0)
        #if DEBUG: print("Weight_Gradient6_Layer3_CH0 : ", len(Weight_Gradient6_Layer3_CH0)) 

        Weight_Gradient7_Layer3_CH0 = Read_DDR(Rd_Address=0x8FB88000,  End_Address=0x8FB9A000)
        Weight_Gradient7_Layer3_CH0_256 = data_32_to_16(Weight_Gradient7_Layer3_CH0)
        #if DEBUG: print("Weight_Gradient7_Layer3_CH0 : ", len(Weight_Gradient7_Layer3_CH0)) 

        Weight_Gradient8_Layer3_CH0 = Read_DDR(Rd_Address=0x8FB9A000,  End_Address=0x8FBAC000)
        Weight_Gradient8_Layer3_CH0_256 = data_32_to_16(Weight_Gradient8_Layer3_CH0)
        #if DEBUG: print("Weight_Gradient8_Layer3_CH0 : ", len(Weight_Gradient8_Layer3_CH0)) 

        Weight_Gradient1_Layer3_CH1 = Read_DDR(Rd_Address=0x9FB1C000,  End_Address=0x9FB2E000)
        Weight_Gradient1_Layer3_CH1_256 = data_32_to_16(Weight_Gradient1_Layer3_CH1)
        #if DEBUG: print("Weight_Gradient1_Layer3_CH1 : ", len(Weight_Gradient1_Layer3_CH1)) 

        Weight_Gradient2_Layer3_CH1 = Read_DDR(Rd_Address=0x9FB2E000,  End_Address=0x9FB40000)
        Weight_Gradient2_Layer3_CH1_256 = data_32_to_16(Weight_Gradient2_Layer3_CH1)
        #if DEBUG: print("Weight_Gradient2_Layer3_CH1 : ", len(Weight_Gradient2_Layer3_CH1)) 

        Weight_Gradient3_Layer3_CH1 = Read_DDR(Rd_Address=0x9FB40000,  End_Address=0x9FB52000)
        Weight_Gradient3_Layer3_CH1_256 = data_32_to_16(Weight_Gradient3_Layer3_CH1)
        #if DEBUG: print("Weight_Gradient3_Layer3_CH1 : ", len(Weight_Gradient3_Layer3_CH1)) 

        Weight_Gradient4_Layer3_CH1 = Read_DDR(Rd_Address=0x9FB52000,  End_Address=0x9FB64000)
        Weight_Gradient4_Layer3_CH1_256 = data_32_to_16(Weight_Gradient4_Layer3_CH1)
        #if DEBUG: print("Weight_Gradient4_Layer3_CH1 : ", len(Weight_Gradient4_Layer3_CH1)) 

        Weight_Gradient5_Layer3_CH1 = Read_DDR(Rd_Address=0x9FB64000,  End_Address=0x9FB76000)
        Weight_Gradient5_Layer3_CH1_256 = data_32_to_16(Weight_Gradient5_Layer3_CH1)
        #if DEBUG: print("Weight_Gradient5_Layer3_CH1 : ", len(Weight_Gradient5_Layer3_CH1)) 

        Weight_Gradient6_Layer3_CH1 = Read_DDR(Rd_Address=0x9FB76000,  End_Address=0x9FB88000)
        Weight_Gradient6_Layer3_CH1_256 = data_32_to_16(Weight_Gradient6_Layer3_CH1)
        #if DEBUG: print("Weight_Gradient6_Layer3_CH1 : ", len(Weight_Gradient6_Layer3_CH1)) 

        Weight_Gradient7_Layer3_CH1 = Read_DDR(Rd_Address=0x9FB88000,  End_Address=0x9FB9A000)
        Weight_Gradient7_Layer3_CH1_256 = data_32_to_16(Weight_Gradient7_Layer3_CH1)
        #if DEBUG: print("Weight_Gradient7_Layer3_CH1 : ", len(Weight_Gradient7_Layer3_CH1)) 

        Weight_Gradient8_Layer3_CH1 = Read_DDR(Rd_Address=0x9FB9A000,  End_Address=0x9FBAC000)
        Weight_Gradient8_Layer3_CH1_256 = data_32_to_16(Weight_Gradient8_Layer3_CH1)
        #if DEBUG: print("Weight_Gradient8_Layer3_CH1 : ", len(Weight_Gradient8_Layer3_CH1)) 
        e = time.time()
        if DEBUG: print("Read WG DDR & 32bit to 16bit : ",e-s)

        '''
        test_out = 'Weight_Result/Weight_Gradient1_Layer3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient1_Layer3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer3_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer3_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer3_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer3_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''

        s = time.time()
        Weight_Gradient1_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer3_CH0_256, Weight_Gradient1_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
        Weight_Gradient2_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer3_CH0_256, Weight_Gradient2_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
        Weight_Gradient3_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer3_CH0_256, Weight_Gradient3_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
        Weight_Gradient4_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer3_CH0_256, Weight_Gradient4_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
        Weight_Gradient5_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer3_CH0_256, Weight_Gradient5_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
        Weight_Gradient6_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer3_CH0_256, Weight_Gradient6_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
        Weight_Gradient7_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer3_CH0_256, Weight_Gradient7_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
        Weight_Gradient8_Layer3 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer3_CH0_256, Weight_Gradient8_Layer3_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=128, In_CH=64, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)
        
        Weight_Gradient_Layer3 = [Weight_Gradient1_Layer3, Weight_Gradient2_Layer3, Weight_Gradient3_Layer3, Weight_Gradient4_Layer3, Weight_Gradient5_Layer3, 
                                Weight_Gradient6_Layer3, Weight_Gradient7_Layer3, Weight_Gradient8_Layer3]
        
        # Weight_Gradient_Layer3 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer3)]  
        Weight_Gradient_Layer3 = list(np.mean(np.array(Weight_Gradient_Layer3), axis=0))
         
        Weight_Gradient_Layer3 = torch.tensor([float(value) for value in Weight_Gradient_Layer3], dtype=torch.float32).reshape(128, 64, 3, 3)   

        Blayer3_end = time.time()
        if DEBUG: print("Layer3 Process Time : ",Blayer3_end-Blayer3_start)

        resume()
        #if DEBUG: print(irq_val)

        #################################################
        #             Backward Layer 2 Start            #
        #################################################
        # check Layer2 IRQ
        Blayer2_start = time.time()
        check_irq_otherlayer()
        s = time.time()
        # self.app_instance .change_color(self.app_instance.L3_IRQ_canvas, self.app_instance.L3_IRQ, "green")
        # Read Gradient of Output After ReLU Backward: 
        Output_Grad1_Layer2_CH0 = Read_DDR(Rd_Address=0x84B00000,  End_Address=0x84B68000)
        Output_Grad1_Layer2_CH0_16 = data_32_to_16(Output_Grad1_Layer2_CH0)

        Output_Grad1_Layer2_CH1 = Read_DDR(Rd_Address=0x94B00000,  End_Address=0x94B68000)
        Output_Grad1_Layer2_CH1_16 = data_32_to_16(Output_Grad1_Layer2_CH1)

        Output_Grad2_Layer2_CH0 = Read_DDR(Rd_Address=0x84B68000,  End_Address=0x84BD0000)
        Output_Grad2_Layer2_CH0_16 = data_32_to_16(Output_Grad2_Layer2_CH0)

        Output_Grad2_Layer2_CH1 = Read_DDR(Rd_Address=0x94B68000,  End_Address=0x94BD0000)
        Output_Grad2_Layer2_CH1_16 = data_32_to_16(Output_Grad2_Layer2_CH1)

        Output_Grad3_Layer2_CH0 = Read_DDR(Rd_Address=0x84BD0000,  End_Address=0x84C38000)
        Output_Grad3_Layer2_CH0_16 = data_32_to_16(Output_Grad3_Layer2_CH0)

        Output_Grad3_Layer2_CH1 = Read_DDR(Rd_Address=0x94BD0000,  End_Address=0x94C38000)
        Output_Grad3_Layer2_CH1_16 = data_32_to_16(Output_Grad3_Layer2_CH1)

        Output_Grad4_Layer2_CH0 = Read_DDR(Rd_Address=0x84C38000,  End_Address=0x84CA0000)
        Output_Grad4_Layer2_CH0_16 = data_32_to_16(Output_Grad4_Layer2_CH0)

        Output_Grad4_Layer2_CH1 = Read_DDR(Rd_Address=0x94C38000,  End_Address=0x94CA0000)
        Output_Grad4_Layer2_CH1_16 = data_32_to_16(Output_Grad4_Layer2_CH1)

        Output_Grad5_Layer2_CH0 = Read_DDR(Rd_Address=0x84CA0000,  End_Address=0x84D08000)
        Output_Grad5_Layer2_CH0_16 = data_32_to_16(Output_Grad5_Layer2_CH0)

        Output_Grad5_Layer2_CH1 = Read_DDR(Rd_Address=0x94CA0000,  End_Address=0x94D08000)
        Output_Grad5_Layer2_CH1_16 = data_32_to_16(Output_Grad5_Layer2_CH1)

        Output_Grad6_Layer2_CH0 = Read_DDR(Rd_Address=0x84D08000,  End_Address=0x84D70000)
        Output_Grad6_Layer2_CH0_16 = data_32_to_16(Output_Grad6_Layer2_CH0)

        Output_Grad6_Layer2_CH1 = Read_DDR(Rd_Address=0x94D08000,  End_Address=0x94D70000)
        Output_Grad6_Layer2_CH1_16 = data_32_to_16(Output_Grad6_Layer2_CH1)

        Output_Grad7_Layer2_CH0 = Read_DDR(Rd_Address=0x84D70000,  End_Address=0x84DD8000)
        Output_Grad7_Layer2_CH0_16 = data_32_to_16(Output_Grad7_Layer2_CH0)

        Output_Grad7_Layer2_CH1 = Read_DDR(Rd_Address=0x94D70000,  End_Address=0x94DD8000)
        Output_Grad7_Layer2_CH1_16 = data_32_to_16(Output_Grad7_Layer2_CH1)

        Output_Grad8_Layer2_CH0 = Read_DDR(Rd_Address=0x84DD8000,  End_Address=0x84E40000)
        Output_Grad8_Layer2_CH0_16 = data_32_to_16(Output_Grad8_Layer2_CH0)

        Output_Grad8_Layer2_CH1 = Read_DDR(Rd_Address=0x94DD8000,  End_Address=0x94E40000)
        Output_Grad8_Layer2_CH1_16 = data_32_to_16(Output_Grad8_Layer2_CH1)
        e = time.time()
        if DEBUG: print("Read OG DDR & 32bit to 16bit : ",e-s)

        s = time.time()
        Output_Grad1_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer2_CH0_16, Output_Grad1_Layer2_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
        Output_Grad2_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer2_CH0_16, Output_Grad2_Layer2_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
        Output_Grad3_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer2_CH0_16, Output_Grad3_Layer2_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
        Output_Grad4_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer2_CH0_16, Output_Grad4_Layer2_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
        Output_Grad5_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer2_CH0_16, Output_Grad5_Layer2_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
        Output_Grad6_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer2_CH0_16, Output_Grad6_Layer2_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
        Output_Grad7_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer2_CH0_16, Output_Grad7_Layer2_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
        Output_Grad8_Layer2 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer2_CH0_16, Output_Grad8_Layer2_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)
        
        Output_Grads_Layer2 = Output_Grad1_Layer2 + Output_Grad2_Layer2 + Output_Grad3_Layer2 + Output_Grad4_Layer2 + \
                                Output_Grad5_Layer2 + Output_Grad6_Layer2 + Output_Grad7_Layer2 + Output_Grad8_Layer2    
        Output_Grad_Layer2 = torch.tensor([float(value) for value in Output_Grads_Layer2], dtype=torch.float32).reshape(8, 32, 104, 104)

        # BReLu Marking
        s = time.time()
        ReLu_Marking1_Layer1_CH0 = Read_DDR(Rd_Address=0x8784C000,  End_Address=0x878B4000)
        ReLu_Marking1_Layer1_CH0_256 = data_32_to_16(ReLu_Marking1_Layer1_CH0)

        ReLu_Marking2_Layer1_CH0 = Read_DDR(Rd_Address=0x878B4000,  End_Address=0x8791C000)
        ReLu_Marking2_Layer1_CH0_256 = data_32_to_16(ReLu_Marking2_Layer1_CH0)

        ReLu_Marking3_Layer1_CH0 = Read_DDR(Rd_Address=0x8791C000,  End_Address=0x87984000)
        ReLu_Marking3_Layer1_CH0_256 = data_32_to_16(ReLu_Marking3_Layer1_CH0)

        ReLu_Marking4_Layer1_CH0 = Read_DDR(Rd_Address=0x87984000,  End_Address=0x879EC000)
        ReLu_Marking4_Layer1_CH0_256 = data_32_to_16(ReLu_Marking4_Layer1_CH0)

        ReLu_Marking5_Layer1_CH0 = Read_DDR(Rd_Address=0x879EC000,  End_Address=0x87A54000)
        ReLu_Marking5_Layer1_CH0_256 = data_32_to_16(ReLu_Marking5_Layer1_CH0)

        ReLu_Marking6_Layer1_CH0 = Read_DDR(Rd_Address=0x87A54000,  End_Address=0x87ABC000)
        ReLu_Marking6_Layer1_CH0_256 = data_32_to_16(ReLu_Marking6_Layer1_CH0)

        ReLu_Marking7_Layer1_CH0 = Read_DDR(Rd_Address=0x87ABC000,  End_Address=0x87B24000)
        ReLu_Marking7_Layer1_CH0_256 = data_32_to_16(ReLu_Marking7_Layer1_CH0)

        ReLu_Marking8_Layer1_CH0 = Read_DDR(Rd_Address=0x87B24000,  End_Address=0x87B8C000)
        ReLu_Marking8_Layer1_CH0_256 = data_32_to_16(ReLu_Marking8_Layer1_CH0)

        ReLu_Marking1_Layer1_CH1 = Read_DDR(Rd_Address=0x9784C000,  End_Address=0x978B4000)
        ReLu_Marking1_Layer1_CH1_256 = data_32_to_16(ReLu_Marking1_Layer1_CH1)

        ReLu_Marking2_Layer1_CH1 = Read_DDR(Rd_Address=0x978B4000,  End_Address=0x9791C000)
        ReLu_Marking2_Layer1_CH1_256 = data_32_to_16(ReLu_Marking2_Layer1_CH1)

        ReLu_Marking3_Layer1_CH1 = Read_DDR(Rd_Address=0x9791C000,  End_Address=0x97984000)
        ReLu_Marking3_Layer1_CH1_256 = data_32_to_16(ReLu_Marking3_Layer1_CH1)

        ReLu_Marking4_Layer1_CH1 = Read_DDR(Rd_Address=0x97984000,  End_Address=0x979EC000)
        ReLu_Marking4_Layer1_CH1_256 = data_32_to_16(ReLu_Marking4_Layer1_CH1)

        ReLu_Marking5_Layer1_CH1 = Read_DDR(Rd_Address=0x979EC000,  End_Address=0x97A54000)
        ReLu_Marking5_Layer1_CH1_256 = data_32_to_16(ReLu_Marking5_Layer1_CH1)

        ReLu_Marking6_Layer1_CH1 = Read_DDR(Rd_Address=0x97A54000,  End_Address=0x97ABC000)
        ReLu_Marking6_Layer1_CH1_256 = data_32_to_16(ReLu_Marking6_Layer1_CH1)

        ReLu_Marking7_Layer1_CH1 = Read_DDR(Rd_Address=0x97ABC000,  End_Address=0x97B24000)
        ReLu_Marking7_Layer1_CH1_256 = data_32_to_16(ReLu_Marking7_Layer1_CH1)

        ReLu_Marking8_Layer1_CH1 = Read_DDR(Rd_Address=0x97B24000,  End_Address=0x97B8C000)
        ReLu_Marking8_Layer1_CH1_256 = data_32_to_16(ReLu_Marking8_Layer1_CH1)
        e= time.time()
        if DEBUG: print("Read RM DDR & 32bit to 16bit : ",e-s)

        # ReLu Reordering
        s = time.time()
        ReLu_Marking1_Layer1 = Read_ReLu_Marking(ReLu_Marking1_Layer1_CH0_256, ReLu_Marking1_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
        ReLu_Marking2_Layer1 = Read_ReLu_Marking(ReLu_Marking2_Layer1_CH0_256, ReLu_Marking2_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
        ReLu_Marking3_Layer1 = Read_ReLu_Marking(ReLu_Marking3_Layer1_CH0_256, ReLu_Marking3_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
        ReLu_Marking4_Layer1 = Read_ReLu_Marking(ReLu_Marking4_Layer1_CH0_256, ReLu_Marking4_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
        ReLu_Marking5_Layer1 = Read_ReLu_Marking(ReLu_Marking5_Layer1_CH0_256, ReLu_Marking5_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
        ReLu_Marking6_Layer1 = Read_ReLu_Marking(ReLu_Marking6_Layer1_CH0_256, ReLu_Marking6_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
        ReLu_Marking7_Layer1 = Read_ReLu_Marking(ReLu_Marking7_Layer1_CH0_256, ReLu_Marking7_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
        ReLu_Marking8_Layer1 = Read_ReLu_Marking(ReLu_Marking8_Layer1_CH0_256, ReLu_Marking8_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, Out_Size=104, Layer8=False)
        e = time.time()
        if DEBUG: print("ReLu Reordering : ",e-s)

        ReLu_Marking_Layer1 = ReLu_Marking1_Layer1 + ReLu_Marking2_Layer1 + ReLu_Marking3_Layer1 + ReLu_Marking4_Layer1 + ReLu_Marking5_Layer1 + \
                                ReLu_Marking6_Layer1 + ReLu_Marking7_Layer1 + ReLu_Marking8_Layer1
        
        ReLu_Marking_Layer1 = torch.tensor([float(value) for value in ReLu_Marking_Layer1], dtype=torch.float32).reshape(8, 32, 104, 104)

        # BReLu Calculate
        # Output_Grad_Layer2_input = torch.tensor(Output_Grad_Layer2, dtype=torch.float32).reshape(8,32,104,104)
        # Layer1_Location = torch.tensor(ReLu_Marking_Layer1, dtype=torch.float32).reshape(8,32,104,104)
        s = time.time()
        relu_mask, location_mask = split_location(ReLu_Marking_Layer1)
        grad_relu_output = backward_active(Output_Grad_Layer2, relu_mask)
        grad_maxpool_output = backward_MaxPool_Location(grad_relu_output, location_mask)
        dL_dgamma_1, dL_dbeta_1, avg_pc_1, backward_const_1 = backward_LightNorm(grad_maxpool_output, layer1_cache)
        e = time.time()
        if DEBUG: print("Software : ",e-s)

        # avg_pc_1 = avg_pc_1.squeeze()
        # backward_const_1 = backward_const_1.squeeze()
        s = time.time()
        avg_pc_1, backward_const_1 = Mean_Var_Dec2Bfloat(avg_pc_1, backward_const_1, Exponent_Bits, Mantissa_Bits)
        e = time.time()
        if DEBUG: print("Dec to Bfloat : ",e-s)

        # Weight_Backward_Layer1 for Soft2Hardware
        s = time.time()
        Weight_Backward_Layer1 = Weight_Hardware_Backward_ReOrdering_OtherLayer(32, 16, data.Weight_Bfloat[1], backward_const_1, avg_pc_1)
        e = time.time()
        if DEBUG: print("Weight Reordering : ",e-s)

        # Break 256To32 and Flip the Data: 
        s = time.time()
        Weight_Backward_Layer1_CH0 = data_256_32(Weight_Backward_Layer1[0])
        Weight_Backward_Layer1_CH1 = data_256_32(Weight_Backward_Layer1[1])
        e = time.time()
        if DEBUG: print("256bit to 32bit : ",e-s)

        # Write Weight For Backward into DDR
        s = time.time()
        Write_DDR(Weight_Backward_Layer1_CH0,Wr_Address=0x823E9000)
        Write_DDR(Weight_Backward_Layer1_CH1,Wr_Address=0x923E9000)
        e = time.time()
        if DEBUG: print("Write DDR : ",e-s)


        # Gradient of Beta Calculation:
        # Beta_Gradient_Layer2 = (Output_Grad_Layer2).sum(dim=(0, 2, 3), keepdim=True)

        # Weight Gradient
        s = time.time()
        Weight_Gradient1_Layer2_CH0 = Read_DDR(Rd_Address=0x8FBAC000,  End_Address=0x8FBB0800)
        Weight_Gradient1_Layer2_CH0_256 = data_32_to_16(Weight_Gradient1_Layer2_CH0)
        #if DEBUG: print("Weight_Gradient1_Layer2_CH0 : ", len(Weight_Gradient1_Layer2_CH0))   

        Weight_Gradient2_Layer2_CH0 = Read_DDR(Rd_Address=0x8FBB0800,  End_Address=0x8FBB5000)
        Weight_Gradient2_Layer2_CH0_256 = data_32_to_16(Weight_Gradient2_Layer2_CH0)
        #if DEBUG: print("Weight_Gradient2_Layer2_CH0 : ", len(Weight_Gradient2_Layer2_CH0))    

        Weight_Gradient3_Layer2_CH0 = Read_DDR(Rd_Address=0x8FBB5000,  End_Address=0x8FBB9800)
        Weight_Gradient3_Layer2_CH0_256 = data_32_to_16(Weight_Gradient3_Layer2_CH0)
        #if DEBUG: print("Weight_Gradient3_Layer2_CH0 : ", len(Weight_Gradient3_Layer2_CH0)) 

        Weight_Gradient4_Layer2_CH0 = Read_DDR(Rd_Address=0x8FBB9800,  End_Address=0x8FBBE000)
        Weight_Gradient4_Layer2_CH0_256 = data_32_to_16(Weight_Gradient4_Layer2_CH0)
        #if DEBUG: print("Weight_Gradient4_Layer2_CH0 : ", len(Weight_Gradient4_Layer2_CH0)) 

        Weight_Gradient5_Layer2_CH0 = Read_DDR(Rd_Address=0x8FBBE000,  End_Address=0x8FBC2800)
        Weight_Gradient5_Layer2_CH0_256 = data_32_to_16(Weight_Gradient5_Layer2_CH0)
        #if DEBUG: print("Weight_Gradient5_Layer2_CH0 : ", len(Weight_Gradient5_Layer2_CH0)) 

        Weight_Gradient6_Layer2_CH0 = Read_DDR(Rd_Address=0x8FBC2800,  End_Address=0x8FBC7000)
        Weight_Gradient6_Layer2_CH0_256 = data_32_to_16(Weight_Gradient6_Layer2_CH0)
        #if DEBUG: print("Weight_Gradient6_Layer2_CH0 : ", len(Weight_Gradient6_Layer2_CH0)) 

        Weight_Gradient7_Layer2_CH0 = Read_DDR(Rd_Address=0x8FBC7000,  End_Address=0x8FBCB800)
        Weight_Gradient7_Layer2_CH0_256 = data_32_to_16(Weight_Gradient7_Layer2_CH0)
        #if DEBUG: print("Weight_Gradient7_Layer2_CH0 : ", len(Weight_Gradient7_Layer2_CH0)) 

        Weight_Gradient8_Layer2_CH0 = Read_DDR(Rd_Address=0x8FBCB800,  End_Address=0x8FBD0000)
        Weight_Gradient8_Layer2_CH0_256 = data_32_to_16(Weight_Gradient8_Layer2_CH0)
        #if DEBUG: print("Weight_Gradient8_Layer2_CH0 : ", len(Weight_Gradient8_Layer2_CH0)) 

        Weight_Gradient1_Layer2_CH1 = Read_DDR(Rd_Address=0x9FBAC000,  End_Address=0x9FBB0800)
        Weight_Gradient1_Layer2_CH1_256 = data_32_to_16(Weight_Gradient1_Layer2_CH1)
        #if DEBUG: print("Weight_Gradient1_Layer2_CH1 : ", len(Weight_Gradient1_Layer2_CH1)) 

        Weight_Gradient2_Layer2_CH1 = Read_DDR(Rd_Address=0x9FBB0800,  End_Address=0x9FBB5000)
        Weight_Gradient2_Layer2_CH1_256 = data_32_to_16(Weight_Gradient2_Layer2_CH1)
        #if DEBUG: print("Weight_Gradient2_Layer2_CH1 : ", len(Weight_Gradient2_Layer2_CH1)) 

        Weight_Gradient3_Layer2_CH1 = Read_DDR(Rd_Address=0x9FBB5000,  End_Address=0x9FBB9800)
        Weight_Gradient3_Layer2_CH1_256 = data_32_to_16(Weight_Gradient3_Layer2_CH1)
        #if DEBUG: print("Weight_Gradient3_Layer2_CH1 : ", len(Weight_Gradient3_Layer2_CH1)) 

        Weight_Gradient4_Layer2_CH1 = Read_DDR(Rd_Address=0x9FBB9800,  End_Address=0x9FBBE000)
        Weight_Gradient4_Layer2_CH1_256 = data_32_to_16(Weight_Gradient4_Layer2_CH1)
        #if DEBUG: print("Weight_Gradient4_Layer2_CH1 : ", len(Weight_Gradient4_Layer2_CH1)) 

        Weight_Gradient5_Layer2_CH1 = Read_DDR(Rd_Address=0x9FBBE000,  End_Address=0x9FBC2800)
        Weight_Gradient5_Layer2_CH1_256 = data_32_to_16(Weight_Gradient5_Layer2_CH1)
        #if DEBUG: print("Weight_Gradient5_Layer2_CH1 : ", len(Weight_Gradient5_Layer2_CH1)) 

        Weight_Gradient6_Layer2_CH1 = Read_DDR(Rd_Address=0x9FBC2800,  End_Address=0x9FBC7000)
        Weight_Gradient6_Layer2_CH1_256 = data_32_to_16(Weight_Gradient6_Layer2_CH1)
        #if DEBUG: print("Weight_Gradient6_Layer2_CH1 : ", len(Weight_Gradient6_Layer2_CH1)) 

        Weight_Gradient7_Layer2_CH1 = Read_DDR(Rd_Address=0x9FBC7000,  End_Address=0x9FBCB800)
        Weight_Gradient7_Layer2_CH1_256 = data_32_to_16(Weight_Gradient7_Layer2_CH1)
        #if DEBUG: print("Weight_Gradient7_Layer2_CH1 : ", len(Weight_Gradient7_Layer2_CH1)) 

        Weight_Gradient8_Layer2_CH1 = Read_DDR(Rd_Address=0x9FBCB800,  End_Address=0x9FBD0000)
        Weight_Gradient8_Layer2_CH1_256 = data_32_to_16(Weight_Gradient8_Layer2_CH1)
        #if DEBUG: print("Weight_Gradient8_Layer2_CH1 : ", len(Weight_Gradient8_Layer2_CH1)) 
        e =time.time()
        if DEBUG: print("Read WG DDR & 32bit to 16bit : ",e-s)

        '''
        test_out = 'Weight_Result/Weight_Gradient1_Layer2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient1_Layer2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer2_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer2_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer2_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer2_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''

        s = time.time()
        Weight_Gradient1_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer2_CH0_256, Weight_Gradient1_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
        Weight_Gradient2_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer2_CH0_256, Weight_Gradient2_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
        Weight_Gradient3_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer2_CH0_256, Weight_Gradient3_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
        Weight_Gradient4_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer2_CH0_256, Weight_Gradient4_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
        Weight_Gradient5_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer2_CH0_256, Weight_Gradient5_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
        Weight_Gradient6_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer2_CH0_256, Weight_Gradient6_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
        Weight_Gradient7_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer2_CH0_256, Weight_Gradient7_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
        Weight_Gradient8_Layer2 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer2_CH0_256, Weight_Gradient8_Layer2_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=64, In_CH=32, Layer8=False)
        e = time.time()
        if DEBUG: print("Bflaot to Dec : ",e-s)
        
        Weight_Gradient_Layer2 = [Weight_Gradient1_Layer2, Weight_Gradient2_Layer2, Weight_Gradient3_Layer2, Weight_Gradient4_Layer2, Weight_Gradient5_Layer2, 
                                Weight_Gradient6_Layer2, Weight_Gradient7_Layer2, Weight_Gradient8_Layer2]
        
        # Weight_Gradient_Layer2 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer2)]   
        Weight_Gradient_Layer2 = list(np.mean(np.array(Weight_Gradient_Layer2), axis=0))
        
        Weight_Gradient_Layer2 = torch.tensor([float(value) for value in Weight_Gradient_Layer2], dtype=torch.float32).reshape(64, 32, 3, 3)   

        Blayer2_end = time.time()
        if DEBUG: print("Layer2 Process Time : ",Blayer2_end-Blayer2_start)

        resume()
        #if DEBUG: print(irq_val)

        #################################################
        #             Backward Layer 1 Start            #
        #################################################
        # check Layer1 IRQ
        Blayer1_start = time.time()
        check_irq_otherlayer()
        s = time.time()
        # self.app_instance .change_color(self.app_instance.L2_IRQ_canvas, self.app_instance.L2_IRQ, "green")
        # Read Gradient of Output After ReLU Backward: 
        Output_Grad1_Layer1_CH0 = Read_DDR(Rd_Address=0x83E00000,  End_Address=0x83ED0000)
        Output_Grad1_Layer1_CH0_16 = data_32_to_16(Output_Grad1_Layer1_CH0)

        Output_Grad1_Layer1_CH1 = Read_DDR(Rd_Address=0x93E00000,  End_Address=0x93ED0000)
        Output_Grad1_Layer1_CH1_16 = data_32_to_16(Output_Grad1_Layer1_CH1)

        Output_Grad2_Layer1_CH0 = Read_DDR(Rd_Address=0x83ED0000,  End_Address=0x83FA0000)
        Output_Grad2_Layer1_CH0_16 = data_32_to_16(Output_Grad2_Layer1_CH0)

        Output_Grad2_Layer1_CH1 = Read_DDR(Rd_Address=0x93ED0000,  End_Address=0x93FA0000)
        Output_Grad2_Layer1_CH1_16 = data_32_to_16(Output_Grad2_Layer1_CH1)

        Output_Grad3_Layer1_CH0 = Read_DDR(Rd_Address=0x83FA0000,  End_Address=0x84070000)
        Output_Grad3_Layer1_CH0_16 = data_32_to_16(Output_Grad3_Layer1_CH0)

        Output_Grad3_Layer1_CH1 = Read_DDR(Rd_Address=0x93FA0000,  End_Address=0x94070000)
        Output_Grad3_Layer1_CH1_16 = data_32_to_16(Output_Grad3_Layer1_CH1)

        Output_Grad4_Layer1_CH0 = Read_DDR(Rd_Address=0x84070000,  End_Address=0x84140000)
        Output_Grad4_Layer1_CH0_16 = data_32_to_16(Output_Grad4_Layer1_CH0)

        Output_Grad4_Layer1_CH1 = Read_DDR(Rd_Address=0x94070000,  End_Address=0x94140000)
        Output_Grad4_Layer1_CH1_16 = data_32_to_16(Output_Grad4_Layer1_CH1)

        Output_Grad5_Layer1_CH0 = Read_DDR(Rd_Address=0x84140000,  End_Address=0x84210000)
        Output_Grad5_Layer1_CH0_16 = data_32_to_16(Output_Grad5_Layer1_CH0)

        Output_Grad5_Layer1_CH1 = Read_DDR(Rd_Address=0x94140000,  End_Address=0x94210000)
        Output_Grad5_Layer1_CH1_16 = data_32_to_16(Output_Grad5_Layer1_CH1)

        Output_Grad6_Layer1_CH0 = Read_DDR(Rd_Address=0x84210000,  End_Address=0x842E0000)
        Output_Grad6_Layer1_CH0_16 = data_32_to_16(Output_Grad6_Layer1_CH0)

        Output_Grad6_Layer1_CH1 = Read_DDR(Rd_Address=0x94210000,  End_Address=0x942E0000)
        Output_Grad6_Layer1_CH1_16 = data_32_to_16(Output_Grad6_Layer1_CH1)

        Output_Grad7_Layer1_CH0 = Read_DDR(Rd_Address=0x842E0000,  End_Address=0x843B0000)
        Output_Grad7_Layer1_CH0_16 = data_32_to_16(Output_Grad7_Layer1_CH0)

        Output_Grad7_Layer1_CH1 = Read_DDR(Rd_Address=0x942E0000,  End_Address=0x943B0000)
        Output_Grad7_Layer1_CH1_16 = data_32_to_16(Output_Grad7_Layer1_CH1)

        Output_Grad8_Layer1_CH0 = Read_DDR(Rd_Address=0x843B0000,  End_Address=0x84480000)
        Output_Grad8_Layer1_CH0_16 = data_32_to_16(Output_Grad8_Layer1_CH0)

        Output_Grad8_Layer1_CH1 = Read_DDR(Rd_Address=0x943B0000,  End_Address=0x94480000)
        Output_Grad8_Layer1_CH1_16 = data_32_to_16(Output_Grad8_Layer1_CH1)
        e = time.time()
        if DEBUG: print("Read OG DDR & 32bit to 16bit : ",e-s)

        s = time.time()
        Output_Grad1_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer1_CH0_16, Output_Grad1_Layer1_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Grad2_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer1_CH0_16, Output_Grad2_Layer1_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Grad3_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer1_CH0_16, Output_Grad3_Layer1_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Grad4_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer1_CH0_16, Output_Grad4_Layer1_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Grad5_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer1_CH0_16, Output_Grad5_Layer1_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Grad6_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer1_CH0_16, Output_Grad6_Layer1_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Grad7_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer1_CH0_16, Output_Grad7_Layer1_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        Output_Grad8_Layer1 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer1_CH0_16, Output_Grad8_Layer1_CH1_16, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)
        
        Output_Grads_Layer1 = Output_Grad1_Layer1 + Output_Grad2_Layer1 + Output_Grad3_Layer1 + Output_Grad4_Layer1 + \
                                Output_Grad5_Layer1 + Output_Grad6_Layer1 + Output_Grad7_Layer1 + Output_Grad8_Layer1    
        Output_Grad_Layer1 = torch.tensor([float(value) for value in Output_Grads_Layer1], dtype=torch.float32).reshape(8, 16, 208, 208)
        
        if DEBUG2 : Save_File(Output_Grad_Layer1, "result/Layer_1_Backward_Input_Gradient")

        # BReLu Marking
        s = time.time()
        ReLu_Marking1_Layer0_CH0 = Read_DDR(Rd_Address=0x871CC000,  End_Address=0x8729C000)
        ReLu_Marking1_Layer0_CH0_256 = data_32_to_16(ReLu_Marking1_Layer0_CH0)

        ReLu_Marking2_Layer0_CH0 = Read_DDR(Rd_Address=0x8729C000,  End_Address=0x8736C000)
        ReLu_Marking2_Layer0_CH0_256 = data_32_to_16(ReLu_Marking2_Layer0_CH0)

        ReLu_Marking3_Layer0_CH0 = Read_DDR(Rd_Address=0x8736C000,  End_Address=0x8743C000)
        ReLu_Marking3_Layer0_CH0_256 = data_32_to_16(ReLu_Marking3_Layer0_CH0)

        ReLu_Marking4_Layer0_CH0 = Read_DDR(Rd_Address=0x8743C000,  End_Address=0x8750C000)
        ReLu_Marking4_Layer0_CH0_256 = data_32_to_16(ReLu_Marking4_Layer0_CH0)

        ReLu_Marking5_Layer0_CH0 = Read_DDR(Rd_Address=0x8750C000,  End_Address=0x875DC000)
        ReLu_Marking5_Layer0_CH0_256 = data_32_to_16(ReLu_Marking5_Layer0_CH0)

        ReLu_Marking6_Layer0_CH0 = Read_DDR(Rd_Address=0x875DC000,  End_Address=0x876AC000)
        ReLu_Marking6_Layer0_CH0_256 = data_32_to_16(ReLu_Marking6_Layer0_CH0)

        ReLu_Marking7_Layer0_CH0 = Read_DDR(Rd_Address=0x876AC000,  End_Address=0x8777C000)
        ReLu_Marking7_Layer0_CH0_256 = data_32_to_16(ReLu_Marking7_Layer0_CH0)

        ReLu_Marking8_Layer0_CH0 = Read_DDR(Rd_Address=0x8777C000,  End_Address=0x8784C000)
        ReLu_Marking8_Layer0_CH0_256 = data_32_to_16(ReLu_Marking8_Layer0_CH0)

        ReLu_Marking1_Layer0_CH1 = Read_DDR(Rd_Address=0x971CC000,  End_Address=0x9729C000)
        ReLu_Marking1_Layer0_CH1_256 = data_32_to_16(ReLu_Marking1_Layer0_CH1)

        ReLu_Marking2_Layer0_CH1 = Read_DDR(Rd_Address=0x9729C000,  End_Address=0x9736C000)
        ReLu_Marking2_Layer0_CH1_256 = data_32_to_16(ReLu_Marking2_Layer0_CH1)

        ReLu_Marking3_Layer0_CH1 = Read_DDR(Rd_Address=0x9736C000,  End_Address=0x9743C000)
        ReLu_Marking3_Layer0_CH1_256 = data_32_to_16(ReLu_Marking3_Layer0_CH1)

        ReLu_Marking4_Layer0_CH1 = Read_DDR(Rd_Address=0x9743C000,  End_Address=0x9750C000)
        ReLu_Marking4_Layer0_CH1_256 = data_32_to_16(ReLu_Marking4_Layer0_CH1)

        ReLu_Marking5_Layer0_CH1 = Read_DDR(Rd_Address=0x9750C000,  End_Address=0x975DC000)
        ReLu_Marking5_Layer0_CH1_256 = data_32_to_16(ReLu_Marking5_Layer0_CH1)

        ReLu_Marking6_Layer0_CH1 = Read_DDR(Rd_Address=0x975DC000,  End_Address=0x976AC000)
        ReLu_Marking6_Layer0_CH1_256 = data_32_to_16(ReLu_Marking6_Layer0_CH1)

        ReLu_Marking7_Layer0_CH1 = Read_DDR(Rd_Address=0x976AC000,  End_Address=0x9777C000)
        ReLu_Marking7_Layer0_CH1_256 = data_32_to_16(ReLu_Marking7_Layer0_CH1)

        ReLu_Marking8_Layer0_CH1 = Read_DDR(Rd_Address=0x9777C000,  End_Address=0x9784C000)
        ReLu_Marking8_Layer0_CH1_256 = data_32_to_16(ReLu_Marking8_Layer0_CH1)
        e = time.time()
        if DEBUG: print("Read RM DDR & 32bit to 16bit : ",e-s)

        # ReLu Reordering
        s = time.time()
        ReLu_Marking1_Layer0 = Read_ReLu_Marking(ReLu_Marking1_Layer0_CH0_256, ReLu_Marking1_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        ReLu_Marking2_Layer0 = Read_ReLu_Marking(ReLu_Marking2_Layer0_CH0_256, ReLu_Marking2_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        ReLu_Marking3_Layer0 = Read_ReLu_Marking(ReLu_Marking3_Layer0_CH0_256, ReLu_Marking3_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        ReLu_Marking4_Layer0 = Read_ReLu_Marking(ReLu_Marking4_Layer0_CH0_256, ReLu_Marking4_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        ReLu_Marking5_Layer0 = Read_ReLu_Marking(ReLu_Marking5_Layer0_CH0_256, ReLu_Marking5_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        ReLu_Marking6_Layer0 = Read_ReLu_Marking(ReLu_Marking6_Layer0_CH0_256, ReLu_Marking6_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        ReLu_Marking7_Layer0 = Read_ReLu_Marking(ReLu_Marking7_Layer0_CH0_256, ReLu_Marking7_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        ReLu_Marking8_Layer0 = Read_ReLu_Marking(ReLu_Marking8_Layer0_CH0_256, ReLu_Marking8_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=208, Layer8=False)
        e = time.time()
        if DEBUG: print("ReLu Reordering : ",e-s)

        ReLu_Marking_Layer0 = ReLu_Marking1_Layer0 + ReLu_Marking2_Layer0 + ReLu_Marking3_Layer0 + ReLu_Marking4_Layer0 + ReLu_Marking5_Layer0 + \
                                ReLu_Marking6_Layer0 + ReLu_Marking7_Layer0 + ReLu_Marking8_Layer0
        
        ReLu_Marking_Layer0 = torch.tensor([float(value) for value in ReLu_Marking_Layer0], dtype=torch.float32).reshape(8, 16, 208, 208)

        # BReLu Calculate
        # Output_Grad_Layer1_input = torch.tensor(Output_Grad_Layer1, dtype=torch.float32).reshape(8,16,208,208)
        # Layer0_Location = torch.tensor(ReLu_Marking_Layer0, dtype=torch.float32).reshape(8,16,208,208)

        s = time.time()
        relu_mask, location_mask = split_location(ReLu_Marking_Layer0)
        grad_relu_output = backward_active(Output_Grad_Layer1, relu_mask)
        # grad_maxpool_output = backward_MaxPool_Location(grad_relu_output, location_mask)
        dL_dgamma_0, dL_dbeta_0, avg_pc_0, backward_const_0 = backward_LightNorm(grad_relu_output, layer0_cache)
        e = time.time()
        if DEBUG: print("Software : ",e-s)

        # avg_pc_0 = avg_pc_0.squeeze()
        # backward_const_0 = backward_const_0.squeeze()
        s = time.time()
        avg_pc_0, backward_const_0 = Mean_Var_Dec2Bfloat(avg_pc_0, backward_const_0, Exponent_Bits, Mantissa_Bits)
        e= time.time()
        if DEBUG: print("Dec to Bfloat : ",e-s)

        # Weight_Backward_Layer0 for Soft2Hardware
        s = time.time()
        Weight_Backward_Layer0 = Weight_Hardware_Backward_ReOrdering_Layer0(16, 16, data.Weight_Bfloat[0], backward_const_0, avg_pc_0)
        e = time.time()
        if DEBUG: print("Weight Reordering : ",e-s)
        #if DEBUG: print("Weight_Backward_Layer0: " + str(len(Weight_Backward_Layer0[0])))
        #if DEBUG: print("Weight_Backward_Layer0: " + str(len(Weight_Backward_Layer0[1])))

        # Break 256To32 and Flip the Data: 
        s = time.time()
        Weight_Backward_Layer0_CH0 = data_256_32(Weight_Backward_Layer0[0])
        Weight_Backward_Layer0_CH1 = data_256_32(Weight_Backward_Layer0[1])
        e = time.time()
        if DEBUG: print("256bit to 32bit : ",e-s)

        # Write Weight For Backward into DDR
        s = time.time()
        Write_DDR(Weight_Backward_Layer0_CH0,Wr_Address=0x823EA400)
        Write_DDR(Weight_Backward_Layer0_CH1,Wr_Address=0x923EA400)
        e = time.time()
        if DEBUG: print("Write DDR : ",e-s)

        # Gradient of Beta Calculation:
        # Beta_Gradient_Layer1 = (Output_Grad_Layer1).sum(dim=(0, 2, 3), keepdim=True)

        # Weight Gradient
        s = time.time()
        Weight_Gradient1_Layer1_CH0 = Read_DDR(Rd_Address=0x8FBD0000,  End_Address=0x8FBD1200)
        Weight_Gradient1_Layer1_CH0_256 = data_32_to_16(Weight_Gradient1_Layer1_CH0)
        #if DEBUG: print("Weight_Gradient1_Layer1_CH0 : ", len(Weight_Gradient1_Layer1_CH0))   

        Weight_Gradient2_Layer1_CH0 = Read_DDR(Rd_Address=0x8FBD1200,  End_Address=0x8FBD2400)
        Weight_Gradient2_Layer1_CH0_256 = data_32_to_16(Weight_Gradient2_Layer1_CH0)
        #if DEBUG: print("Weight_Gradient2_Layer1_CH0 : ", len(Weight_Gradient2_Layer1_CH0))    

        Weight_Gradient3_Layer1_CH0 = Read_DDR(Rd_Address=0x8FBD2400,  End_Address=0x8FBD3600)
        Weight_Gradient3_Layer1_CH0_256 = data_32_to_16(Weight_Gradient3_Layer1_CH0)
        #if DEBUG: print("Weight_Gradient3_Layer1_CH0 : ", len(Weight_Gradient3_Layer1_CH0)) 

        Weight_Gradient4_Layer1_CH0 = Read_DDR(Rd_Address=0x8FBD3600,  End_Address=0x8FBD4800)
        Weight_Gradient4_Layer1_CH0_256 = data_32_to_16(Weight_Gradient4_Layer1_CH0)
        #if DEBUG: print("Weight_Gradient4_Layer1_CH0 : ", len(Weight_Gradient4_Layer1_CH0)) 

        Weight_Gradient5_Layer1_CH0 = Read_DDR(Rd_Address=0x8FBD4800,  End_Address=0x8FBD5A00)
        Weight_Gradient5_Layer1_CH0_256 = data_32_to_16(Weight_Gradient5_Layer1_CH0)
        #if DEBUG: print("Weight_Gradient5_Layer1_CH0 : ", len(Weight_Gradient5_Layer1_CH0)) 

        Weight_Gradient6_Layer1_CH0 = Read_DDR(Rd_Address=0x8FBD5A00,  End_Address=0x8FBD6C00)
        Weight_Gradient6_Layer1_CH0_256 = data_32_to_16(Weight_Gradient6_Layer1_CH0)
        #if DEBUG: print("Weight_Gradient6_Layer1_CH0 : ", len(Weight_Gradient6_Layer1_CH0)) 

        Weight_Gradient7_Layer1_CH0 = Read_DDR(Rd_Address=0x8FBD6C00,  End_Address=0x8FBD7E00)
        Weight_Gradient7_Layer1_CH0_256 = data_32_to_16(Weight_Gradient7_Layer1_CH0)
        #if DEBUG: print("Weight_Gradient7_Layer1_CH0 : ", len(Weight_Gradient7_Layer1_CH0)) 

        Weight_Gradient8_Layer1_CH0 = Read_DDR(Rd_Address=0x8FBD7E00,  End_Address=0x8FBD9000)
        Weight_Gradient8_Layer1_CH0_256 = data_32_to_16(Weight_Gradient8_Layer1_CH0)
        #if DEBUG: print("Weight_Gradient8_Layer1_CH0 : ", len(Weight_Gradient8_Layer1_CH0)) 

        Weight_Gradient1_Layer1_CH1 = Read_DDR(Rd_Address=0x9FBD0000,  End_Address=0x9FBD1200)
        Weight_Gradient1_Layer1_CH1_256 = data_32_to_16(Weight_Gradient1_Layer1_CH1)
        #if DEBUG: print("Weight_Gradient1_Layer1_CH1 : ", len(Weight_Gradient1_Layer1_CH1)) 

        Weight_Gradient2_Layer1_CH1 = Read_DDR(Rd_Address=0x9FBD1200,  End_Address=0x9FBD2400)
        Weight_Gradient2_Layer1_CH1_256 = data_32_to_16(Weight_Gradient2_Layer1_CH1)
        #if DEBUG: print("Weight_Gradient2_Layer1_CH1 : ", len(Weight_Gradient2_Layer1_CH1)) 

        Weight_Gradient3_Layer1_CH1 = Read_DDR(Rd_Address=0x9FBD2400,  End_Address=0x9FBD3600)
        Weight_Gradient3_Layer1_CH1_256 = data_32_to_16(Weight_Gradient3_Layer1_CH1)
        #if DEBUG: print("Weight_Gradient3_Layer1_CH1 : ", len(Weight_Gradient3_Layer1_CH1)) 

        Weight_Gradient4_Layer1_CH1 = Read_DDR(Rd_Address=0x9FBD3600,  End_Address=0x9FBD4800)
        Weight_Gradient4_Layer1_CH1_256 = data_32_to_16(Weight_Gradient4_Layer1_CH1)
        #if DEBUG: print("Weight_Gradient4_Layer1_CH1 : ", len(Weight_Gradient4_Layer1_CH1)) 

        Weight_Gradient5_Layer1_CH1 = Read_DDR(Rd_Address=0x9FBD4800,  End_Address=0x9FBD5A00)
        Weight_Gradient5_Layer1_CH1_256 = data_32_to_16(Weight_Gradient5_Layer1_CH1)
        #if DEBUG: print("Weight_Gradient5_Layer1_CH1 : ", len(Weight_Gradient5_Layer1_CH1)) 

        Weight_Gradient6_Layer1_CH1 = Read_DDR(Rd_Address=0x9FBD5A00,  End_Address=0x9FBD6C00)
        Weight_Gradient6_Layer1_CH1_256 = data_32_to_16(Weight_Gradient6_Layer1_CH1)
        #if DEBUG: print("Weight_Gradient6_Layer1_CH1 : ", len(Weight_Gradient6_Layer1_CH1)) 

        Weight_Gradient7_Layer1_CH1 = Read_DDR(Rd_Address=0x9FBD6C00,  End_Address=0x9FBD7E00)
        Weight_Gradient7_Layer1_CH1_256 = data_32_to_16(Weight_Gradient7_Layer1_CH1)
        #if DEBUG: print("Weight_Gradient7_Layer1_CH1 : ", len(Weight_Gradient7_Layer1_CH1)) 

        Weight_Gradient8_Layer1_CH1 = Read_DDR(Rd_Address=0x9FBD7E00,  End_Address=0x9FBD9000)
        Weight_Gradient8_Layer1_CH1_256 = data_32_to_16(Weight_Gradient8_Layer1_CH1)
        #if DEBUG: print("Weight_Gradient8_Layer1_CH1 : ", len(Weight_Gradient8_Layer1_CH1)) 
        e = time.time()
        if DEBUG: print("Read WG DDR & 32bit to 16bit : ",e-s)

        '''
        test_out = 'Weight_Result/Weight_Gradient1_Layer1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient1_Layer1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer1_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer1_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer1_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer1_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''

        s = time.time()
        Weight_Gradient1_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient1_Layer1_CH0_256, Weight_Gradient1_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient2_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient2_Layer1_CH0_256, Weight_Gradient2_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient3_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient3_Layer1_CH0_256, Weight_Gradient3_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient4_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient4_Layer1_CH0_256, Weight_Gradient4_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient5_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient5_Layer1_CH0_256, Weight_Gradient5_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient6_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient6_Layer1_CH0_256, Weight_Gradient6_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient7_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient7_Layer1_CH0_256, Weight_Gradient7_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        Weight_Gradient8_Layer1 = Read_WeightGradient_Bfloat2Dec(Weight_Gradient8_Layer1_CH0_256, Weight_Gradient8_Layer1_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=32, In_CH=16, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)
        
        Weight_Gradient_Layer1 = [Weight_Gradient1_Layer1, Weight_Gradient2_Layer1, Weight_Gradient3_Layer1, Weight_Gradient4_Layer1, Weight_Gradient5_Layer1, 
                                Weight_Gradient6_Layer1, Weight_Gradient7_Layer1, Weight_Gradient8_Layer1]
        
        # Weight_Gradient_Layer1 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer1)]   
        Weight_Gradient_Layer1 = list(np.mean(np.array(Weight_Gradient_Layer1), axis=0))
        
        Weight_Gradient_Layer1 = torch.tensor([float(value) for value in Weight_Gradient_Layer1], dtype=torch.float32).reshape(32, 16, 3, 3)   
        
        if DEBUG2 : Save_File(Weight_Gradient_Layer1, "result/Layer_1_Backward_Weight_Gradient")

        Blayer1_end = time.time()
        if DEBUG: print("Layer1 Process Time : ",Blayer1_end-Blayer1_start)

        resume()

        #################################################
        #             Backward Layer 0 Start            #
        #################################################
        # check Layer0 IRQ
        Blayer0_start = time.time()
        check_irq_otherlayer()
        # self.app_instance .change_color(self.app_instance.L1_IRQ_canvas, self.app_instance.L1_IRQ, "green")
        '''
        # Read Gradient of Output After ReLU Backward: 
        Output_Grad1_Layer0_CH0 = Read_DDR(Rd_Address=0x9384000,  End_Address=0x96C4000)
        Output_Grad1_Layer0_CH0 = data_32_to_16(Output_Grad1_Layer0_CH0)

        Output_Grad1_Layer0_CH1 = Read_DDR(Rd_Address=0x19384000,  End_Address=0x196C4000)
        Output_Grad1_Layer0_CH1 = data_32_to_16(Output_Grad1_Layer0_CH1)

        Output_Grad2_Layer0_CH0 = Read_DDR(Rd_Address=0x96C4000,  End_Address=0x9A04000)
        Output_Grad2_Layer0_CH0 = data_32_to_16(Output_Grad2_Layer0_CH0)

        Output_Grad2_Layer0_CH1 = Read_DDR(Rd_Address=0x196C4000,  End_Address=0x19A04000)
        Output_Grad2_Layer0_CH1 = data_32_to_16(Output_Grad2_Layer0_CH1)

        Output_Grad3_Layer0_CH0 = Read_DDR(Rd_Address=0x9A04000,  End_Address=0x9D44000)
        Output_Grad3_Layer0_CH0 = data_32_to_16(Output_Grad3_Layer0_CH0)

        Output_Grad3_Layer0_CH1 = Read_DDR(Rd_Address=0x19A04000,  End_Address=0x19D44000)
        Output_Grad3_Layer0_CH1 = data_32_to_16(Output_Grad3_Layer0_CH1)

        Output_Grad4_Layer0_CH0 = Read_DDR(Rd_Address=0x9D44000,  End_Address=0xA084000)
        Output_Grad4_Layer0_CH0 = data_32_to_16(Output_Grad4_Layer0_CH0)

        Output_Grad4_Layer0_CH1 = Read_DDR(Rd_Address=0x19D44000,  End_Address=0x1A084000)
        Output_Grad4_Layer0_CH1 = data_32_to_16(Output_Grad4_Layer0_CH1)

        Output_Grad5_Layer0_CH0 = Read_DDR(Rd_Address=0xA084000,  End_Address=0xA3C4000)
        Output_Grad5_Layer0_CH0 = data_32_to_16(Output_Grad5_Layer0_CH0)

        Output_Grad5_Layer0_CH1 = Read_DDR(Rd_Address=0x1A084000,  End_Address=0x1A3C4000)
        Output_Grad5_Layer0_CH1 = data_32_to_16(Output_Grad5_Layer0_CH1)

        Output_Grad6_Layer0_CH0 = Read_DDR(Rd_Address=0xA3C4000,  End_Address=0xA704000)
        Output_Grad5_Layer0_CH1 = data_32_to_16(Output_Grad5_Layer0_CH1)

        Output_Grad6_Layer0_CH1 = Read_DDR(Rd_Address=0x1A3C4000,  End_Address=0x1A704000)
        Output_Grad6_Layer0_CH1 = data_32_to_16(Output_Grad6_Layer0_CH1)

        Output_Grad7_Layer0_CH0 = Read_DDR(Rd_Address=0xA704000,  End_Address=0xAA44000)
        Output_Grad7_Layer0_CH0 = data_32_to_16(Output_Grad7_Layer0_CH0)

        Output_Grad7_Layer0_CH1 = Read_DDR(Rd_Address=0x1A704000,  End_Address=0x1AA44000)
        Output_Grad7_Layer0_CH1 = data_32_to_16(Output_Grad7_Layer0_CH1)

        Output_Grad8_Layer0_CH0 = Read_DDR(Rd_Address=0xAA44000,  End_Address=0xAD84000)
        Output_Grad8_Layer0_CH0 = data_32_to_16(Output_Grad8_Layer0_CH0)

        Output_Grad8_Layer0_CH1 = Read_DDR(Rd_Address=0x1AA44000,  End_Address=0x1AD84000)
        Output_Grad8_Layer0_CH1 = data_32_to_16(Output_Grad8_Layer0_CH1)
        


        Output_Grad1_Layer0 = Read_OutFmap_Bfloat2Dec(Output_Grad1_Layer0_CH0, Output_Grad1_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)
        Output_Grad2_Layer0 = Read_OutFmap_Bfloat2Dec(Output_Grad2_Layer0_CH0, Output_Grad2_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)
        Output_Grad3_Layer0 = Read_OutFmap_Bfloat2Dec(Output_Grad3_Layer0_CH0, Output_Grad3_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)
        Output_Grad4_Layer0 = Read_OutFmap_Bfloat2Dec(Output_Grad4_Layer0_CH0, Output_Grad4_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)
        Output_Grad5_Layer0 = Read_OutFmap_Bfloat2Dec(Output_Grad5_Layer0_CH0, Output_Grad5_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)
        Output_Grad6_Layer0 = Read_OutFmap_Bfloat2Dec(Output_Grad6_Layer0_CH0, Output_Grad6_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)
        Output_Grad7_Layer0 = Read_OutFmap_Bfloat2Dec(Output_Grad7_Layer0_CH0, Output_Grad7_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)
        Output_Grad8_Layer0 = Read_OutFmap_Bfloat2Dec(Output_Grad8_Layer0_CH0, Output_Grad8_Layer0_CH1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)
        Output_Grads_Layer0 = Output_Grad1_Layer0 + Output_Grad2_Layer0 + Output_Grad3_Layer0 + Output_Grad4_Layer0 + \
                                Output_Grad5_Layer0 + Output_Grad6_Layer0 + Output_Grad7_Layer0 + Output_Grad8_Layer0    
        Output_Grad_Layer0 = torch.tensor([float(value) for value in Output_Grads_Layer0], dtype=torch.float32).reshape(8, 16, 416, 416)
        
        '''


        # Gradient of Beta Calculation:
        # Beta_Gradient_Layer0 = (Output_Grad_Layer0).sum(dim=(0, 2, 3), keepdim=True)

        # Weight Gradient
        s = time.time()
        Weight_Gradient1_Layer0_CH0 = Read_DDR(Rd_Address=0x8FBD9000,  End_Address=0x8FBD9900)
        Weight_Gradient1_Layer0_CH0_256 = data_32_to_16(Weight_Gradient1_Layer0_CH0)
        #if DEBUG: print("Weight_Gradient1_Layer0_CH0 : ", len(Weight_Gradient1_Layer0_CH0))   

        Weight_Gradient2_Layer0_CH0 = Read_DDR(Rd_Address=0x8FBD9900,  End_Address=0x8FBDA200)
        Weight_Gradient2_Layer0_CH0_256 = data_32_to_16(Weight_Gradient2_Layer0_CH0)
        #if DEBUG: print("Weight_Gradient2_Layer0_CH0 : ", len(Weight_Gradient2_Layer0_CH0))    

        Weight_Gradient3_Layer0_CH0 = Read_DDR(Rd_Address=0x8FBDA200,  End_Address=0x8FBDAB00)
        Weight_Gradient3_Layer0_CH0_256 = data_32_to_16(Weight_Gradient3_Layer0_CH0)
        #if DEBUG: print("Weight_Gradient3_Layer0_CH0 : ", len(Weight_Gradient3_Layer0_CH0)) 

        Weight_Gradient4_Layer0_CH0 = Read_DDR(Rd_Address=0x8FBDAB00,  End_Address=0x8FBDB400)
        Weight_Gradient4_Layer0_CH0_256 = data_32_to_16(Weight_Gradient4_Layer0_CH0)
        #if DEBUG: print("Weight_Gradient4_Layer0_CH0 : ", len(Weight_Gradient4_Layer0_CH0)) 

        Weight_Gradient5_Layer0_CH0 = Read_DDR(Rd_Address=0x8FBDB400,  End_Address=0x8FBDBD00)
        Weight_Gradient5_Layer0_CH0_256 = data_32_to_16(Weight_Gradient5_Layer0_CH0)
        #if DEBUG: print("Weight_Gradient5_Layer0_CH0 : ", len(Weight_Gradient5_Layer0_CH0)) 

        Weight_Gradient6_Layer0_CH0 = Read_DDR(Rd_Address=0x8FBDBD00,  End_Address=0x8FBDC600)
        Weight_Gradient6_Layer0_CH0_256 = data_32_to_16(Weight_Gradient6_Layer0_CH0)
        #if DEBUG: print("Weight_Gradient6_Layer0_CH0 : ", len(Weight_Gradient6_Layer0_CH0)) 

        Weight_Gradient7_Layer0_CH0 = Read_DDR(Rd_Address=0x8FBDC600,  End_Address=0x8FBDCF00)
        Weight_Gradient7_Layer0_CH0_256 = data_32_to_16(Weight_Gradient7_Layer0_CH0)
        #if DEBUG: print("Weight_Gradient7_Layer0_CH0 : ", len(Weight_Gradient7_Layer0_CH0)) 

        Weight_Gradient8_Layer0_CH0 = Read_DDR(Rd_Address=0x8FBDCF00,  End_Address=0x8FBDD800)
        Weight_Gradient8_Layer0_CH0_256 = data_32_to_16(Weight_Gradient8_Layer0_CH0)
        #if DEBUG: print("Weight_Gradient8_Layer0_CH0 : ", len(Weight_Gradient8_Layer0_CH0)) 

        Weight_Gradient1_Layer0_CH1 = Read_DDR(Rd_Address=0x9FBD9000,  End_Address=0x9FBD9900)
        Weight_Gradient1_Layer0_CH1_256 = data_32_to_16(Weight_Gradient1_Layer0_CH1)
        #if DEBUG: print("Weight_Gradient1_Layer0_CH1 : ", len(Weight_Gradient1_Layer0_CH1)) 

        Weight_Gradient2_Layer0_CH1 = Read_DDR(Rd_Address=0x9FBD9900,  End_Address=0x9FBDA200)
        Weight_Gradient2_Layer0_CH1_256 = data_32_to_16(Weight_Gradient2_Layer0_CH1)
        #if DEBUG: print("Weight_Gradient2_Layer0_CH1 : ", len(Weight_Gradient2_Layer0_CH1)) 

        Weight_Gradient3_Layer0_CH1 = Read_DDR(Rd_Address=0x9FBDA200,  End_Address=0x9FBDAB00)
        Weight_Gradient3_Layer0_CH1_256 = data_32_to_16(Weight_Gradient3_Layer0_CH1)
        #if DEBUG: print("Weight_Gradient3_Layer0_CH1 : ", len(Weight_Gradient3_Layer0_CH1)) 

        Weight_Gradient4_Layer0_CH1 = Read_DDR(Rd_Address=0x9FBDAB00,  End_Address=0x9FBDB400)
        Weight_Gradient4_Layer0_CH1_256 = data_32_to_16(Weight_Gradient4_Layer0_CH1)
        #if DEBUG: print("Weight_Gradient4_Layer0_CH1 : ", len(Weight_Gradient4_Layer0_CH1)) 

        Weight_Gradient5_Layer0_CH1 = Read_DDR(Rd_Address=0x9FBDB400,  End_Address=0x9FBDBD00)
        Weight_Gradient5_Layer0_CH1_256 = data_32_to_16(Weight_Gradient5_Layer0_CH1)
        #if DEBUG: print("Weight_Gradient5_Layer0_CH1 : ", len(Weight_Gradient5_Layer0_CH1)) 

        Weight_Gradient6_Layer0_CH1 = Read_DDR(Rd_Address=0x9FBDBD00,  End_Address=0x9FBDC600)
        Weight_Gradient6_Layer0_CH1_256 = data_32_to_16(Weight_Gradient6_Layer0_CH1)
        #if DEBUG: print("Weight_Gradient6_Layer0_CH1 : ", len(Weight_Gradient6_Layer0_CH1)) 

        Weight_Gradient7_Layer0_CH1 = Read_DDR(Rd_Address=0x9FBDC600,  End_Address=0x9FBDCF00)
        Weight_Gradient7_Layer0_CH1_256 = data_32_to_16(Weight_Gradient7_Layer0_CH1)
        #if DEBUG: print("Weight_Gradient7_Layer0_CH1 : ", len(Weight_Gradient7_Layer0_CH1)) 

        Weight_Gradient8_Layer0_CH1 = Read_DDR(Rd_Address=0x9FBDCF00,  End_Address=0x9FBDD800)
        Weight_Gradient8_Layer0_CH1_256 = data_32_to_16(Weight_Gradient8_Layer0_CH1)
        #if DEBUG: print("Weight_Gradient8_Layer0_CH1 : ", len(Weight_Gradient8_Layer0_CH1)) 
        e = time.time()
        if DEBUG: print("Read WG DDR & 32bit to 16bit : ",e-s)

        '''
        test_out = 'Weight_Result/Weight_Gradient1_Layer0_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer0_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient1_Layer0_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient1_Layer0_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer0_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer0_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient2_Layer0_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient2_Layer0_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer0_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer0_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient3_Layer0_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient3_Layer0_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer0_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer0_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient4_Layer0_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient4_Layer0_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer0_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer0_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient5_Layer0_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient5_Layer0_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer0_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer0_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient6_Layer0_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient6_Layer0_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer0_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer0_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient7_Layer0_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient7_Layer0_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer0_CH0.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer0_CH0:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()

        test_out = 'Weight_Result/Weight_Gradient8_Layer0_CH1.txt'
        with open(test_out, 'w+') as test_output:
            for item in Weight_Gradient8_Layer0_CH1:
                line = str(item) 
                test_output.write(line + '\n')   
        test_output.close()
        '''

        s = time.time()
        Weight_Gradient1_Layer0 = Read_WeightGradient_Bfloat2Dec_Layer0(Weight_Gradient1_Layer0_CH0_256, Weight_Gradient1_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, In_CH=16, Layer8=False)
        Weight_Gradient2_Layer0 = Read_WeightGradient_Bfloat2Dec_Layer0(Weight_Gradient2_Layer0_CH0_256, Weight_Gradient2_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, In_CH=16, Layer8=False)
        Weight_Gradient3_Layer0 = Read_WeightGradient_Bfloat2Dec_Layer0(Weight_Gradient3_Layer0_CH0_256, Weight_Gradient3_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, In_CH=16, Layer8=False)
        Weight_Gradient4_Layer0 = Read_WeightGradient_Bfloat2Dec_Layer0(Weight_Gradient4_Layer0_CH0_256, Weight_Gradient4_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, In_CH=16, Layer8=False)
        Weight_Gradient5_Layer0 = Read_WeightGradient_Bfloat2Dec_Layer0(Weight_Gradient5_Layer0_CH0_256, Weight_Gradient5_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, In_CH=16, Layer8=False)
        Weight_Gradient6_Layer0 = Read_WeightGradient_Bfloat2Dec_Layer0(Weight_Gradient6_Layer0_CH0_256, Weight_Gradient6_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, In_CH=16, Layer8=False)
        Weight_Gradient7_Layer0 = Read_WeightGradient_Bfloat2Dec_Layer0(Weight_Gradient7_Layer0_CH0_256, Weight_Gradient7_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, In_CH=16, Layer8=False)
        Weight_Gradient8_Layer0 = Read_WeightGradient_Bfloat2Dec_Layer0(Weight_Gradient8_Layer0_CH0_256, Weight_Gradient8_Layer0_CH1_256, Exponent_Bits, Mantissa_Bits, Out_CH=16, In_CH=16, Layer8=False)
        e = time.time()
        if DEBUG: print("Bfloat to Dec : ",e-s)
        
        s = time.time()
        Weight_Gradient_Layer0 = [Weight_Gradient1_Layer0, Weight_Gradient2_Layer0, Weight_Gradient3_Layer0, Weight_Gradient4_Layer0, Weight_Gradient5_Layer0, 
                                Weight_Gradient6_Layer0, Weight_Gradient7_Layer0, Weight_Gradient8_Layer0]
        
        Weight_Gradient_Layer0 = [sum(map(float, item)) / len(item) for item in zip(*Weight_Gradient_Layer0)]   
        # Weight_Gradient_Layer0 = list(np.mean(np.array(Weight_Gradient_Layer0), axis=0))
        
        Weight_Gradient_Layer0 = torch.tensor([float(value) for value in Weight_Gradient_Layer0], dtype=torch.float32).reshape(16, 3, 3, 3)   
        e = time.time()
        if DEBUG: print("reshape : ",e-s)

        # Gradient Value for Weight Update
        Weight_Gradient = [Weight_Gradient_Layer0, Weight_Gradient_Layer1, Weight_Gradient_Layer2, Weight_Gradient_Layer3, Weight_Gradient_Layer4,
                        Weight_Gradient_Layer5, Weight_Gradient_Layer6, Weight_Gradient_Layer7, Weight_Gradient_Layer8]
        
        Beta_Gradient = [dL_dbeta_0, dL_dbeta_1, dL_dbeta_2, dL_dbeta_3, dL_dbeta_4, dL_dbeta_5,
                        dL_dbeta_6, dL_dbeta_7]
        
        Gamma_Gradient = [dL_dgamma_0, dL_dgamma_1, dL_dgamma_2, dL_dgamma_3, dL_dgamma_4,
                        dL_dgamma_5, dL_dgamma_6, dL_dgamma_7]

        Blayer1_end = time.time()
        if DEBUG: print("Layer0 Process Time : ",Blayer1_end-Blayer1_start)

        # pdb.set_trace() 
        resume()
        
        
        if DEBUG: print("Weight_Gradient : ", Weight_Gradient[0][0])

        if DEBUG: print("Bias_Grad   : ", Bias_Grad[0])

        if DEBUG: print("Beta_Gradient   : ", Beta_Gradient[0])

        if DEBUG: print("Gamma_Gradient  : ", Gamma_Gradient[0])
        
        # import pdb
        # pdb.set_trace
        
        return Weight_Gradient, Bias_Grad, Beta_Gradient, Gamma_Gradient
    

    def Write_Weight(self, data):
        # Pre-Processing Class Initialization
        # global Weight_Bfloat, Bias_Bfloat, Beta_Bfloat, Gamma_Bfloat, Running_Mean_Bfloat, Running_Var_Bfloat

        data.Weight_Bfloat, data.Bias_Bfloat, data.Beta_Bfloat, data.Gamma_Bfloat, data.Running_Mean_Bfloat, data.Running_Var_Bfloat = self.PreProcessing.Weight_Converted_Func(\
            data.Weight_Dec, data.Bias_Dec, data.Beta_Dec, data.Gamma_Dec, data.Running_Mean_Dec, data.Running_Var_Dec)  
        
        if DEBUG: print("Weight_Dec : ", data.Weight_Dec[0][0])
        if DEBUG: print("Weight_Dec : ", data.Weight_Bfloat[0][0])

        if DEBUG: print("Bias_Dec   : ", data.Bias_Dec[0])
        if DEBUG: print("Bias_Dec   : ", data.Bias_Bfloat[0])

        if DEBUG: print("Beta_Dec   : ", data.Beta_Dec[0])
        if DEBUG: print("Beta_Dec   : ", data.Beta_Bfloat[0])

        if DEBUG: print("Gamma_Dec  : ", data.Gamma_Dec[0])
        if DEBUG: print("Gamma_Dec  : ", data.Gamma_Bfloat[0])

        
        
        s = time.time()
        Weight_1st_Layer0 = New_Weight_Hardware_ReOrdering_Layer0(16,       16,   data.Weight_Bfloat[0], ['0000']*16, ['0000']*16, ['0000']*16, Iteration="1")
        Weight_1st_Layer1 = New_Weight_Hardware_ReOrdering_OtherLayer(  32, 16,   data.Weight_Bfloat[1], ['0000']*32, ['0000']*32, ['0000']*32, Iteration="1")
        Weight_1st_Layer2 = New_Weight_Hardware_ReOrdering_OtherLayer ( 64, 32,   data.Weight_Bfloat[2], ['0000']*64, ['0000']*64, ['0000']*64, Iteration="1")
        Weight_1st_Layer3 = New_Weight_Hardware_ReOrdering_OtherLayer( 128, 64,   data.Weight_Bfloat[3], ['0000']*128, ['0000']*128, ['0000']*128, Iteration="1")
        Weight_1st_Layer4 = New_Weight_Hardware_ReOrdering_OtherLayer( 256, 128,  data.Weight_Bfloat[4], ['0000']*256, ['0000']*256, ['0000']*256, Iteration="1")
        Weight_1st_Layer5 = New_Weight_Hardware_ReOrdering_OtherLayer( 512, 256,  data.Weight_Bfloat[5], ['0000']*512, ['0000']*512, ['0000']*512, Iteration="1")
        Weight_1st_Layer6 = New_Weight_Hardware_ReOrdering_OtherLayer(1024, 512,  data.Weight_Bfloat[6], ['0000']*1024, ['0000']*1024, ['0000']*1024, Iteration="1")
        Weight_1st_Layer7 = New_Weight_Hardware_ReOrdering_OtherLayer(1024, 1024, data.Weight_Bfloat[7], ['0000']*1024, ['0000']*1024, ['0000']*1024, Iteration="1")
        Weight_1st_Layer8 = New_Weight_Hardware_ReOrdering_Layer8(     128, 1024, data.Weight_Bfloat[8], data.Bias_Bfloat)
        e = time.time()
        if DEBUG: print("Weight Ordering : ",e-s)
        
        # List for Each DDR Channels: 
        Weight_1st_CH0 = Weight_1st_Layer0[0] + Weight_1st_Layer1[0] + Weight_1st_Layer2[0] + Weight_1st_Layer3[0] + Weight_1st_Layer4[0] + \
                        Weight_1st_Layer5[0] + Weight_1st_Layer6[0] + Weight_1st_Layer7[0] + Weight_1st_Layer8[0]
        Weight_1st_CH1 = Weight_1st_Layer0[1] + Weight_1st_Layer1[1] + Weight_1st_Layer2[1] + Weight_1st_Layer3[1] + Weight_1st_Layer4[1] + \
                        Weight_1st_Layer5[1] + Weight_1st_Layer6[1] + Weight_1st_Layer7[1] + Weight_1st_Layer8[1]
        
        # Break 256To32 and Flip the Data: 
        s = time.time()
        Weight_1st_CH0 = data_256_32(Weight_1st_CH0)
        Weight_1st_CH1 = data_256_32(Weight_1st_CH1)
        e = time.time()
        if DEBUG: print("256bit to 32bit Convert :",e-s) 
                        
        # Write Weights into DDR: 
        s = time.time()
        Write_DDR(Weight_1st_CH0, Wr_Address=0x80000000)
        Write_DDR(Weight_1st_CH1, Wr_Address=0x90000000)    
        e = time.time()
        if DEBUG: print("Write Weight : ",e-s)


    def Write_Image(self, data):
        # Pre-Processing Class Initialization
        # global Weight_Bfloat, Bias_Bfloat, Beta_Bfloat, Gamma_Bfloat, Running_Mean_Bfloat, Running_Var_Bfloat

        s = time.time()
        image = self.PreProcessing.Image_Converted_Func(data.im_data)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(data.im_data[0].permute(1,2,0).numpy().astype(np.uint8))
        # plt.show()
        Image1 = Fmap_Hardware_ReOrdering_Layer0(16, image[0])
        Image2 = Fmap_Hardware_ReOrdering_Layer0(16, image[1])
        Image3 = Fmap_Hardware_ReOrdering_Layer0(16, image[2])
        Image4 = Fmap_Hardware_ReOrdering_Layer0(16, image[3])
        Image5 = Fmap_Hardware_ReOrdering_Layer0(16, image[4])
        Image6 = Fmap_Hardware_ReOrdering_Layer0(16, image[5])
        Image7 = Fmap_Hardware_ReOrdering_Layer0(16, image[6])
        Image8 = Fmap_Hardware_ReOrdering_Layer0(16, image[7])
        e = time.time()
        if DEBUG: print("Fmap Ordering : ",e-s)
        
        Images_CH0 = Image1[0] + Image2[0] + Image3[0] + Image4[0] + Image5[0] + Image6[0] + Image7[0] + Image8[0]
        Images_CH1 = Image1[1] + Image2[1] + Image3[1] + Image4[1] + Image5[1] + Image6[1] + Image7[1] + Image8[1]
            
        # Break 256To32 and Flip the Data: 
        s = time.time()
        Images_CH0 = data_256_32(Images_CH0)
        Images_CH1 = data_256_32(Images_CH1)
        e = time.time()
        if DEBUG: print("256bit to 32 bit Convert : ",e-s)
       
        # Write Images into DDR: 
        s = time.time()
        Write_DDR(Images_CH0, Wr_Address=0x82400000)
        Write_DDR(Images_CH1, Wr_Address=0x92400000)  
        e = time.time()
        if DEBUG: print("Write Image : ",e-s)
        
        
        Im0_ch0 = Read_DDR(Rd_Address=0x82400000, End_Address=0X82740000)
        Im0_ch0 = data_32_to_16(Im0_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Im0_ch1 = Read_DDR(Rd_Address=0x92400000, End_Address=0X92740000)
        Im0_ch1 = data_32_to_16(Im0_ch1)
        
        Image0 = Read_OutFmap_Bfloat2Dec(Im0_ch0, Im0_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)
        # if DEBUG: print(len(Image0))

        # Image0 = torch.tensor([float(value) for value in Image0], dtype=torch.float32).reshape(1, 16, 416, 416)
        
        Im1_ch0 = Read_DDR(Rd_Address=0X82740000, End_Address=0X82A80000)
        Im1_ch0 = data_32_to_16(Im1_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Im1_ch1 = Read_DDR(Rd_Address=0x92740000, End_Address=0X92A80000)
        Im1_ch1 = data_32_to_16(Im1_ch1)
        
        Image1 = Read_OutFmap_Bfloat2Dec(Im1_ch0, Im1_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)

        # Image1 = torch.tensor([float(value) for value in Image1], dtype=torch.float32).reshape(1, 16, 416, 416)
        
        Im2_ch0 = Read_DDR(Rd_Address=0X82A80000, End_Address=0X82DC0000)
        Im2_ch0 = data_32_to_16(Im2_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Im2_ch1 = Read_DDR(Rd_Address=0X92A80000, End_Address=0X92DC0000)
        Im2_ch1 = data_32_to_16(Im2_ch1)
        
        Image2 = Read_OutFmap_Bfloat2Dec(Im2_ch0, Im2_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)

        # Image2 = torch.tensor([float(value) for value in Image2], dtype=torch.float32).reshape(1, 16, 416, 416)
        
        Im3_ch0 = Read_DDR(Rd_Address=0X82DC0000, End_Address=0X83100000)
        Im3_ch0 = data_32_to_16(Im3_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Im3_ch1 = Read_DDR(Rd_Address=0X92DC0000, End_Address=0X93100000)
        Im3_ch1 = data_32_to_16(Im3_ch1)
        
        Image3 = Read_OutFmap_Bfloat2Dec(Im3_ch0, Im3_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)

        # Image3 = torch.tensor([float(value) for value in Image3], dtype=torch.float32).reshape(1, 16, 416, 416)
        
        Im4_ch0 = Read_DDR(Rd_Address=0X83100000, End_Address=0X83440000)
        Im4_ch0 = data_32_to_16(Im4_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Im4_ch1 = Read_DDR(Rd_Address=0X93100000, End_Address=0X93440000)
        Im4_ch1 = data_32_to_16(Im4_ch1)
        
        Image4 = Read_OutFmap_Bfloat2Dec(Im4_ch0, Im4_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)

        # Image4 = torch.tensor([float(value) for value in Image4], dtype=torch.float32).reshape(1, 16, 416, 416)
        
        Im5_ch0 = Read_DDR(Rd_Address=0X83440000, End_Address=0X83780000)
        Im5_ch0 = data_32_to_16(Im5_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Im5_ch1 = Read_DDR(Rd_Address=0X93440000, End_Address=0X93780000)
        Im5_ch1 = data_32_to_16(Im5_ch1)
        
        Image5 = Read_OutFmap_Bfloat2Dec(Im5_ch0, Im5_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)

        # Image5 = torch.tensor([float(value) for value in Image5], dtype=torch.float32).reshape(1, 16, 416, 416)
        
        Im6_ch0 = Read_DDR(Rd_Address=0X83780000, End_Address=0X83AC0000)
        Im6_ch0 = data_32_to_16(Im6_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Im6_ch1 = Read_DDR(Rd_Address=0X93780000, End_Address=0X93AC0000)
        Im6_ch1 = data_32_to_16(Im6_ch1)
        
        Image6 = Read_OutFmap_Bfloat2Dec(Im6_ch0, Im6_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)

        # Image6 = torch.tensor([float(value) for value in Image6], dtype=torch.float32).reshape(1, 16, 416, 416)
        
        Im7_ch0 = Read_DDR(Rd_Address=0X83AC0000, End_Address=0X83E00000)
        Im7_ch0 = data_32_to_16(Im7_ch0)
        #if DEBUG: print("ch1 image 7 : ", len(Layer0_1st_Iter_Image7_CH1))

        Im7_ch1 = Read_DDR(Rd_Address=0X93AC0000, End_Address=0X93E00000)
        Im7_ch1 = data_32_to_16(Im7_ch1)
        
        Image7 = Read_OutFmap_Bfloat2Dec(Im7_ch0, Im7_ch1, Exponent_Bits, Mantissa_Bits, Out_CH=16, Out_Size=416, Layer8=False)

        # Image7 = torch.tensor([float(value) for value in Image7], dtype=torch.float32).reshape(1, 16, 416, 416)
        
        # Image = Image0 + Image1 + Image2 + Image3 + Image4 + Image5 + Image6 + Image7
        Image0 = Image0[0:3*1*416*416]
        Image1 = Image1[0:3*1*416*416]
        Image2 = Image2[0:3*1*416*416]
        Image3 = Image3[0:3*1*416*416]
        Image4 = Image4[0:3*1*416*416]
        Image5 = Image5[0:3*1*416*416]
        Image6 = Image6[0:3*1*416*416]
        Image7 = Image7[0:3*1*416*416]
        
        
        Image0 = torch.tensor([float(value) for value in Image0], dtype=torch.float32).reshape(1, 3, 416, 416)
        Image1 = torch.tensor([float(value) for value in Image1], dtype=torch.float32).reshape(1, 3, 416, 416)
        Image2 = torch.tensor([float(value) for value in Image2], dtype=torch.float32).reshape(1, 3, 416, 416)
        Image3 = torch.tensor([float(value) for value in Image3], dtype=torch.float32).reshape(1, 3, 416, 416)
        Image4 = torch.tensor([float(value) for value in Image4], dtype=torch.float32).reshape(1, 3, 416, 416)
        Image5 = torch.tensor([float(value) for value in Image5], dtype=torch.float32).reshape(1, 3, 416, 416)
        Image6 = torch.tensor([float(value) for value in Image6], dtype=torch.float32).reshape(1, 3, 416, 416)
        Image7 = torch.tensor([float(value) for value in Image7], dtype=torch.float32).reshape(1, 3, 416, 416)

        _Inputs = [Image0, Image1, Image2, Image3, Image4, Image5, Image6, Image7 ]
        if DEBUG2 : Save_File(_Inputs, "result/Input_Data")
        
        # save_txt("Input_Image", Image, module="Conv", layer_no=0, save_txt=True, phase="Forward")
        
        


    def Write_Image_Test(self):
        # Pre-Processing Class Initialization
        # global Weight_Bfloat, Bias_Bfloat, Beta_Bfloat, Gamma_Bfloat, Running_Mean_Bfloat, Running_Var_Bfloat

        image = self.PreProcessing.Image_Converted_Func_Test(self.Image)
        Image1 = Fmap_Hardware_ReOrdering_Layer0(16, image[0])
        Image2 = Fmap_Hardware_ReOrdering_Layer0(16, image[1])
        Image3 = Fmap_Hardware_ReOrdering_Layer0(16, image[2])
        Image4 = Fmap_Hardware_ReOrdering_Layer0(16, image[3])
        Image5 = Fmap_Hardware_ReOrdering_Layer0(16, image[4])
        Image6 = Fmap_Hardware_ReOrdering_Layer0(16, image[5])
        Image7 = Fmap_Hardware_ReOrdering_Layer0(16, image[6])
        Image8 = Fmap_Hardware_ReOrdering_Layer0(16, image[7])
            
        # Break 256To32 and Flip the Data: 
        Image1_CH0 = data_256_32(Image1[0])
        Image1_CH1 = data_256_32(Image1[1])

        # Break 256To32 and Flip the Data: 
        Image2_CH0 = data_256_32(Image2[0])
        Image2_CH1 = data_256_32(Image2[1])

        # Break 256To32 and Flip the Data: 
        Image3_CH0 = data_256_32(Image3[0])
        Image3_CH1 = data_256_32(Image3[1])

        # Break 256To32 and Flip the Data: 
        Image4_CH0 = data_256_32(Image4[0])
        Image4_CH1 = data_256_32(Image4[1])

        # Break 256To32 and Flip the Data: 
        Image5_CH0 = data_256_32(Image5[0])
        Image5_CH1 = data_256_32(Image5[1])

        # Break 256To32 and Flip the Data: 
        Image6_CH0 = data_256_32(Image6[0])
        Image6_CH1 = data_256_32(Image6[1])

        # Break 256To32 and Flip the Data: 
        Image7_CH0 = data_256_32(Image7[0])
        Image7_CH1 = data_256_32(Image7[1])

        # Break 256To32 and Flip the Data: 
        Image8_CH0 = data_256_32(Image8[0])
        Image8_CH1 = data_256_32(Image8[1])
        
        # Write Images into DDR: 
        Write_DDR(Image1_CH0, Wr_Address=0x81200000)
        Write_DDR(Image1_CH1, Wr_Address=0x91200000)  

        # Write Images into DDR: 
        Write_DDR(Image2_CH0, Wr_Address=0x81B00000)
        Write_DDR(Image2_CH1, Wr_Address=0x91B00000) 

        # Write Images into DDR: 
        Write_DDR(Image3_CH0, Wr_Address=0x81E40000)
        Write_DDR(Image3_CH1, Wr_Address=0x91E40000) 

        # Write Images into DDR: 
        Write_DDR(Image4_CH0, Wr_Address=0x82180000)
        Write_DDR(Image4_CH1, Wr_Address=0x92180000) 

        # Write Images into DDR: 
        Write_DDR(Image5_CH0, Wr_Address=0x824C0000)
        Write_DDR(Image5_CH1, Wr_Address=0x924C0000) 

        # Write Images into DDR: 
        Write_DDR(Image6_CH0, Wr_Address=0x82800000)
        Write_DDR(Image6_CH1, Wr_Address=0x92800000) 

        # Write Images into DDR: 
        Write_DDR(Image7_CH0, Wr_Address=0x82B40000)
        Write_DDR(Image7_CH1, Wr_Address=0x92B40000) 

        # Write Images into DDR: 
        Write_DDR(Image8_CH0, Wr_Address=0x82E80000)
        Write_DDR(Image8_CH1, Wr_Address=0x92E80000) 

