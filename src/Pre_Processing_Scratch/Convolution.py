import torch
import time
import pickle
import math
import numpy as np
from numba import njit, prange
import ctypes

# Define ANSI escape codes for text formatting
GREEN = '\033[92m'  # Green color
RED   = '\033[91m'  # Red color
RESET = '\033[0m'   # Reset color to default

colors = [
    '\033[35m',  # Purple
    '\033[36m',  # Cyan
    '\033[33m',  # Yellow
    '\033[34m',  # Blue
    '\033[31m',  # Red
    '\033[32m',  # Green
    '\033[93m',  # Light Yellow
    '\033[94m'   # Light Blue
    '\033[91m',  # Light Red
    '\033[92m',  # Light Green
]
libconv = ctypes.CDLL('E:/RESEARCH/Convolution_OPT/x64/Debug/CONVOLUTION_OPT.dll')
# Define the argument and result types
# libconv.conv2d.argtypes = [
#     ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), # N, C, H, W, input
#     ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), # F, HH, WW, kernel
#     ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int                # output, pad, stride
# ]
class Python_Conv(object):
    @staticmethod
    def Forward(x, w, conv_param, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        out = None
        pad = conv_param['pad']
        stride = conv_param['stride']
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        H_out = int(1 + (H + 2 * pad - HH) / stride)
        W_out = int(1 + (W + 2 * pad - WW) / stride)
       
        _curr_time = time.time()
        out = torch.zeros((N, F, H_out, W_out), dtype=x.dtype, device=x.device)
        
        # Flatten the arrays and get pointers to their data
        input_ptr = x.flatten().contiguous().data_ptr()
        kernel_ptr = w.flatten().contiguous().data_ptr()
        output_ptr = out.flatten().contiguous().data_ptr()
        
        libconv.conv2d(N, C, H, W, ctypes.cast(input_ptr, ctypes.POINTER(ctypes.c_float)),
               F, HH, WW, ctypes.cast(kernel_ptr, ctypes.POINTER(ctypes.c_float)),
               ctypes.cast(output_ptr, ctypes.POINTER(ctypes.c_float)),
               pad, stride)
        
        out = out.reshape(N, F, H_out, W_out)


        # print(output_array)
        # out = torch.zeros((N, F, H_out, W_out), dtype=x.dtype, device=x.device)
        # x = torch.nn.functional.pad(x, (pad, pad, pad, pad))
        
        # _curr_time = time.time()
        # for n in range(N):
        #     for f in range(F):
        #         for height in range(H_out):
        #             for width in range(W_out):
        #                 out[n, f, height, width] = (x[n, :, height * stride:height * stride + HH, width * stride:width * stride + WW] * w[f]).sum()
                  
        _time= (time.time() - _curr_time)  
        print("Time taken",_time)
 
        cache = (x, w, conv_param)
        # print(f"\n\nAccumulation Result Default:{out}")
        # print(out.shape, output_array.shape)

        return out, cache
    
    
    @staticmethod
    def Forward_fast(x, w, conv_param, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        pad = conv_param['pad']
        stride = conv_param['stride']
        N, A, H, W = x.shape  
        w_reshape = w.permute(1, 0, 2, 3)
        w_flipped = torch.flip(w_reshape, dims=(2, 3))
        F, C, HH, WW = w_flipped.shape
        H_out = int(1 + (H + 2 * pad - HH) / stride)
        W_out = int(1 + (W + 2 * pad - WW) / stride)
        x = torch.nn.functional.pad(x, (pad, pad, pad, pad))
        out = torch.zeros((N, C, H_out + 2, W_out +2), dtype=x.dtype, device=x.device)
        _curr_time = time.time()
        for n in range(N):
            for f in range(F):
                for height in range(H_out):
                    for width in range(W_out):
                        out[n, :, height * stride:height * stride + HH, width * stride:width * stride + WW] += w_flipped[f] \
                        * x[n, f, height * stride:height * stride + HH, width * stride:width * stride + WW]            
        out = out[:, :, 1:-1, 1:-1]  # delete padded "pixels"
        # out= out[:, :, :-2, :-2]


        print("OUT RESULT: \n ", out)
        print("INPUT Shape: ", x.shape)
        print("WEIGHT Shape: ", w.shape)    
        print("OUTPUT Shape: ", out.shape) 
        _time= (time.time() - _curr_time)  
        print("Time taken",_time)
        cache = (x, w, conv_param)

        return out, cache
    







    @staticmethod
    def Batch_Normalization(x, gamma, beta, layer_no=[], save_txt=False, save_hex=False, phase=[], args = None):
        
        out, cache = None, None
   
        eps = 1e-5
        D = gamma.shape[0]
        num_chunks = 1
        # running_mean = bn_params.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
        # running_var = bn_params.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))
        
        running_mean = torch.tensor([-0.2920,  0.6815,  0.0352,  0.0316, -0.0126, -0.0390, -0.0013, -0.2882, -0.2624,  0.0064, -0.0118,  0.0088, -0.0653, -0.0112, -0.0088, -0.0602])

        running_var =  torch.tensor([-0.2920,  0.6815,  0.0352,  0.0316, -0.0126, -0.0390, -0.0013, -0.2882, -0.2624,  0.0064, -0.0118,  0.0088, -0.0653, -0.0112, -0.0088, -0.0602])
        
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
## ---------------------------------------------------------------------------------------------------------- ##

        # output = (x - avg) * scale # This is first calculation
        # output = output * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1) # this is second calculation
        avg =  torch.tensor([-0.2920,  0.6815,  0.0352,  0.0316, -0.0126, -0.0390, -0.0013, -0.2882, -0.2624,  0.0064, -0.0118,  0.0088, -0.0653, -0.0112, -0.0088, -0.0602]).reshape(1, -1, 1, 1)
        output = gamma.view(1, -1, 1, 1)* scale # This is first calculation
        output = output*(x - avg)  
        output2= output + beta.view(1, -1, 1, 1) # this is second calculation
        
        
        

## ---------------------------------------------------------------------------------------------------------- ##
        
        # ctx.save_for_backward(X, gamma, beta, output, scale)
        
            
        running_mean = running_mean * momentum + (1 - momentum) * avg
        running_var = running_var * momentum + (1 - momentum) * scale
        
        cache = (x, gamma, beta, output, scale, scale_fix, avg, avg_max, avg_min, eps, num_chunks, max_index, min_index)
        
        # Subtraction: Mean 
        Sub = avg
        
        # Multiplication: scale * gamma
        scale_reshape = scale.squeeze()
        Mul = scale_reshape * gamma
        
        # Addition: Beta
        Add = beta
    
    
        
        return output2, cache
    
    @staticmethod
    # def backward(x, w, layer_no=[], save_txt=False, save_hex=False, phase=[]):
    #     dout = w
    #     pad = 0
    #     stride = 1
        
    def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        x, w,b, conv_param = cache
        pad = conv_param['pad']
        stride = conv_param['stride']
        N, F, H_dout, W_dout = dout.shape
        F, C, HH, WW = w.shape
        dx, dw = None, None
        dw = torch.zeros_like(w)
        dx = torch.zeros_like(x)
        for n in range(N):
            for f in range(F):
                for height in range(H_dout):
                    for width in range(W_dout):
                        dw[f] += x[n, :, height * stride:height * stride + HH, width * stride:width * stride + WW] * dout[n, f, height, width]
                        dx[n, :, height * stride:height * stride + HH, width * stride:width * stride + WW] += w[f] * dout[ n, f, height, width]
        dx = dx[:, :, 1:-1, 1:-1]  # delete padded "pixels"

        return dx, dw
    @staticmethod
    def backward_slow(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        x, w,b, conv_param = cache
        pad = conv_param['pad']
        stride = conv_param['stride']
        
        reshaped_w = w.permute(1, 0, 2, 3)
        w_flipped = torch.flip(reshaped_w, dims=(2, 3))
        F, C, HH, WW = w_flipped.shape
        N, C, H, W = dout.shape
        H_dout = int(1 + (H + 2 * pad - HH) / stride)
        W_dout = int(1 + (W + 2 * pad - WW) / stride)
        dout = torch.nn.functional.pad(dout, (pad, pad, pad, pad))
        
        dout= dout.reshape(N, C, H_dout, W_dout)
        dx, dw = None, None
        dx = torch.zeros_like(x)
        dw = torch.zeros_like(w)

        for n in range(N):
            for f in range(F):
                for height in range(H_dout):
                    for width in range(W_dout):
                        dx[n, f, height, width] = (dout[n, :, height * stride:height * stride + HH, width * stride:width * stride + WW] * w_flipped[f]).sum()
        dx = dx[:, :, 1:-1, 1:-1]  # delete padded "pixels"
        return dx, dw
    
    @staticmethod
    def backward_CC(dout, x, w):
        pad = 1
        stride = 1
        reshaped_w = w.permute(1, 0, 2, 3)
        w_flipped = torch.flip(reshaped_w, dims=(2, 3))
        F, C, HH, WW = w_flipped.shape
        N, C, H, W = dout.shape
        H_dout = int(1 + (H + 2 * pad - HH) / stride)
        W_dout = int(1 + (W + 2 * pad - WW) / stride)
        dout_pad =  torch.nn.functional.pad(dout, (pad, pad, pad, pad))
        dx, dw = None, None
        dw = torch.zeros_like(w)
        dx = torch.zeros_like(x)
        x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad))        
        for n in range(N):
            for f in range(F):
                for height in range(H_dout):
                    for width in range(W_dout):
                        dx[n, f, height, width] = (dout_pad[n, :, height * stride:height * stride + HH, width * stride:width * stride + WW] * w_flipped[f]).sum()
        
        # for f in range(F):
        #     for c in range(C):
        #         for height in range(HH):
        #             for width in range(WW):
        #                 # Convolve dout with the relevant region from x to compute dw
        #                 dw[f, c, height, width] = (dout[:, f, :, :] * x_padded[:, c, height:height + H_dout, width:width + W_dout]).sum()

        # dx = dx[:, :, 1:-1, 1:-1]  # delete padded "pixels"
        return dx, dw
    
    @njit(parallel=True)
    def Forward_numba_optimized(x, w, conv_param):
        pad = conv_param['pad']
        stride = conv_param['stride']
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        H_out = int(1 + (H + 2 * pad - HH) / stride)
        W_out = int(1 + (W + 2 * pad - WW) / stride)

        # Pad the input
        x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))

        # Initialize output array
        out = np.zeros((N, F, H_out, W_out), dtype=x.dtype)

        # Perform convolution
        for n in prange(N):
            for f in prange(F):
                for height in prange(H_out):
                    for width in prange(W_out):
                        x_slice = x_padded[n, :, height * stride:height * stride + HH, width * stride:width * stride + WW]
                        out[n, f, height, width] = np.sum(x_slice * w[f])
        cache = (x, w, conv_param)
        return out, cache


class Torch_FastConv(object):

    @staticmethod
    def Forward(x, w, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad, bias=False)
        layer.weight = torch.nn.Parameter(w)
        # layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        
        cache = (x, w, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            # db = layer.bias.grad.detach()
            layer.weight.grad = None
                      
        except RuntimeError:
            dx, dw = torch.zeros_like(tx), torch.zeros_like(layer.weight)
        return dx, dw

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
            
def train():
    
    with open('Temp_Files/Python/Forward_cache.pickle', 'rb') as handle:
        cache = pickle.load(handle)


    print("Loading previous files for Loss Calculation.")
    with open('Temp_Files/Python/loss.pickle', 'rb') as handle:
        loss = pickle.load(handle)
    with open('Temp_Files/Python/loss_gradients.pickle', 'rb') as handle:
        loss_grad = pickle.load(handle)
            
    return  loss_grad, cache

if __name__ == "__main__":
    prefix = "b"
    Image_Path = f"./{prefix}_image.txt"
    Weight_Path =f'./{prefix}_weight.txt'
    processing_start_time = time.time()
    if prefix=='s':
        # Image = torch.tensor(ReadTXT2Tensor(Image_Path)).reshape(1, 1, 3, 3)
        Image = ReadTXT2Tensor(Image_Path).reshape(1, 3, 5, 5)
        Weight =ReadTXT2Tensor(Weight_Path).reshape(16, 3, 3, 3)
    else:
        Image = ReadTXT2Tensor(Image_Path).reshape(8, 512, 13, 13)
        Weight =ReadTXT2Tensor(Weight_Path).reshape(1024, 512, 3, 3)
        dout = ReadTXT2Tensor(f'./{prefix}_dout.txt').reshape(8, 1024, 13, 13)
        Gamma = ReadTXT2Tensor(f'./{prefix}_gamma.txt').reshape(16)
        Beta = ReadTXT2Tensor(f'./{prefix}_beta.txt').reshape(16)
    processing_end_time = time.time()
    
    processing_time = processing_end_time - processing_start_time 
    # print(f"\n\nProcessing Time: {processing_time}")
    # # Convolution: 
    conv_param = {'stride': 1, 'pad': 1}
    device = "cpu"
    # device = 'cuda:0'
    # Conv_Result, cache = Python_Conv.Forward(Image, Weight, conv_param, device)  
    # processing_end_time = time.time()
    # processing_time = processing_end_time - processing_start_time 
    # # print(f"\n\nProcessing Time: {processing_time}")
    # Conv_Result_Path = "./Conv_FW_FAST.txt"
    # save_file(Conv_Result, Conv_Result_Path)
    
    
    # loss_grad, cache= train() 
    # lDout, grads = Python_Conv.backward(loss_grad, cache['8'])
    
    # lDout, grads = Python_Conv.backward_CC(dout, Image, Weight)
    # Conv_Result_Path = "./Conv_FW_FAST.txt"
    # save_file(lDout, Conv_Result_Path)
    # Conv_Result_Path = "./back_conv_slow.txt"
    # save_file(lDout, Conv_Result_Path)
    
    # an, bn_cache = Python_Conv.Batch_Normalization(
    #                                                     Image,
    #                                                     Gamma,
    #                                                     Beta,
    #                                                     layer_no=0,
    #                                                     save_txt=True,
    #                                                     save_hex=False,
    #                                                     phase=0, 
    #                                                     args = 0,
    #                                                     )
    
    # BN_PATH = "./BN_AVG2.txt"
    # save_file(an, BN_PATH)
    
    