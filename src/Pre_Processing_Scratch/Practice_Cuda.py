import torch
import time
import ctypes
import os
libconv = ctypes.CDLL('./Practice_cuda.so')
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def forward(x, w, conv_param, device):
    x = x.to(device)
    w = w.to(device)

    pad = conv_param['pad']
    stride = conv_param['stride']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    out = torch.zeros((N, F, H_out, W_out), dtype=x.dtype, device= device)
    
    if device == "cuda":
        _curr_time = time.time()
        input_ptr = x.flatten().contiguous().data_ptr()
        kernel_ptr = w.flatten().contiguous().data_ptr()
        output_ptr = out.flatten().contiguous().data_ptr()

        libconv.conv2d(N, C, H, W, ctypes.cast(input_ptr, ctypes.POINTER(ctypes.c_float)),
                F, HH, WW, ctypes.cast(kernel_ptr, ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(output_ptr, ctypes.POINTER(ctypes.c_float)),
                pad, stride)
        _time= (time.time() - _curr_time) 
        print("Time taken by CUDA Program: ",_time)
    else:
        _curr_time = time.time()
        x = torch.nn.functional.pad(x, (pad, pad, pad, pad))
        for n in range(N):
            for f in range(F):
                for height in range(H_out):
                    for width in range(W_out):
                        out[n, f, height, width] = (x[n, :, height * stride:height * stride + HH, width * stride:width * stride + WW] * w[f]).sum()
        _time= (time.time() - _curr_time) 
        print("Time taken by CPU is: ",_time)
 
    print("OUT RESULT: \n ", out)
    cache = (x, w, conv_param)
    

    return out, cache

def ReadTXT2Tensor(path):
    read = open(path, mode="r")
    read_data = read.readlines()
    Read_Data = []
    for value in read_data[1:]:
        Read_Data.append(float(value.replace("\n", "")))
    Read_Data = torch.tensor(Read_Data)
    return Read_Data

def save_file(weights, file_path):
    weights = weights.flatten().tolist()
    with open(file_path, 'w') as file:
        for weight in weights:
            file.write(f"{weight}\n")

if __name__ == "__main__":
    prefix = "s"
    Image_Path = f"./{prefix}_image.txt"
    Weight_Path =f'./{prefix}_weight.txt'
    processing_start_time = time.time()
    if prefix=='s':
        Image = ReadTXT2Tensor(Image_Path).reshape(1, 1, 5, 5)
        Weight =ReadTXT2Tensor(Weight_Path).reshape(1, 1, 3, 3)
    else:
        Image = ReadTXT2Tensor(Image_Path).reshape(8, 512, 13, 13)
        Weight =ReadTXT2Tensor(Weight_Path).reshape(1024, 512, 3, 3)
    processing_end_time = time.time()
    
    device = "cpu"
    conv_param = {'stride': 1, 'pad': 1}
    Conv_Result, cache = forward(Image, Weight, conv_param,  device)