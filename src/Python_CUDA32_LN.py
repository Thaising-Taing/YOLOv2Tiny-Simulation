from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import warnings
import torch
from pathlib import Path
import math
import time
import ctypes
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.simplefilter("ignore", UserWarning)
libconv = ctypes.CDLL('convolution_cuda.so')
os.environ["CUDA_VISIBLE_DEVICES"] = '3'


# Python_Convolution without Bias
class Python_Conv(object):

    @staticmethod
    def forward(x, w, conv_param):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        w = w.to(device)
        out = None
        # pad = conv_param['pad']
        pad = 1
        stride = conv_param['stride']
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        H_out = int(1 + (H + 2 * pad - HH) / stride)
        W_out = int(1 + (W + 2 * pad - WW) / stride)
       
        _curr_time = time.time()
        out = torch.zeros((N, F, H_out, W_out), dtype=x.dtype, device= "cuda")
        input_ptr = x.flatten().contiguous().data_ptr()
        kernel_ptr = w.flatten().contiguous().data_ptr()
        output_ptr = out.flatten().contiguous().data_ptr()

        libconv.conv2d(N, C, H, W, ctypes.cast(input_ptr, ctypes.POINTER(ctypes.c_float)),
               F, HH, WW, ctypes.cast(kernel_ptr, ctypes.POINTER(ctypes.c_float)),
               ctypes.cast(output_ptr, ctypes.POINTER(ctypes.c_float)),
               pad, stride)
        out = out.reshape(N, F, H_out, W_out)
        _time= (time.time() - _curr_time)  
        # print("Time taken by Layer is: ",_time)
        cache = (x, w, conv_param)
        

        return out, cache

    @staticmethod
    def backward(dout, cache):
        x, w, conv_param = cache
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dout = dout.to(device)
        x = x.to(device)
        w = w.to(device)
        pad = 1 
        stride = conv_param['stride']
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape

        dout_gpu = dout.contiguous()
        x_gpu = x.contiguous()
        w_gpu = w.contiguous()

        dx = torch.zeros_like(x_gpu)
        dw = torch.zeros_like(w_gpu)

        x_ptr = x_gpu.flatten().data_ptr()
        w_ptr = w_gpu.flatten().data_ptr()
        dout_ptr = dout_gpu.flatten().data_ptr()
        dw_ptr = dw.flatten().data_ptr()
        dx_ptr = dx.flatten().data_ptr()
        _curr_time = time.time()

        libconv.conv2d_backward_dw(
            N, C, H, W, ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float)), 
            F, HH, WW,ctypes.cast(dout_ptr, ctypes.POINTER(ctypes.c_float)), ctypes.cast(dw_ptr,ctypes.POINTER(ctypes.c_float)), H, W, stride, pad)
           
        reshaped_w = w.permute(1, 0, 2, 3)
        w_flipped = torch.flip(reshaped_w, dims=(2, 3))
        FF, CC, HH, WW = w_flipped.shape
        w_transpose = w_flipped.contiguous()
        w_ptr_transpose = w_transpose.flatten().data_ptr()

        libconv.conv2d(N, F, H, W, ctypes.cast(dout_ptr, ctypes.POINTER(ctypes.c_float)),
                    FF, HH, WW, ctypes.cast(w_ptr_transpose, ctypes.POINTER(ctypes.c_float)),
                    ctypes.cast(dx_ptr, ctypes.POINTER(ctypes.c_float)), stride, pad)

        
        _time = (time.time() - _curr_time)  
        # print("Time taken by Backward Layer is:", _time)
        

        dx = dx.reshape(N, C, H, W)
     
        return dx, dw 


# Python_Convolution with Bias
class Python_ConvB(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        out = None
        pad = conv_param['pad']
        stride = conv_param['stride']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        w = w.to(device)
        b = b.to(device)
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        H_out = int(1 + (H + 2 * pad - HH) / stride)
        W_out = int(1 + (W + 2 * pad - WW) / stride)
        
        _curr_time = time.time()
        # Flatten the arrays and get pointers to their data
        out = torch.zeros((N, F, H_out, W_out), dtype=x.dtype, device= "cuda")
        input_ptr  = x.flatten().contiguous().data_ptr()
        kernel_ptr = w.flatten().contiguous().data_ptr()
        output_ptr = out.flatten().contiguous().data_ptr()
        bias_ptr   = b.flatten().contiguous().data_ptr()

        libconv.conv2d_WB(N, C, H, W, ctypes.cast(input_ptr, ctypes.POINTER(ctypes.c_float)),
               F, HH, WW, ctypes.cast(kernel_ptr, ctypes.POINTER(ctypes.c_float)),
               ctypes.cast(bias_ptr, ctypes.POINTER(ctypes.c_float)),
               ctypes.cast(output_ptr, ctypes.POINTER(ctypes.c_float)),
               pad, stride)
        out = out.reshape(N, F, H_out, W_out)
                          
        _time= (time.time() - _curr_time)  
        # print("Time taken by Layer WB","is: ",_time)
        cache = (x, w, b, conv_param)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, w, b, conv_param = cache
        dx, dw_bias, db = None, None, None
        w = w.to(device)
        b = b.to(device)

        pad = conv_param['pad']
        stride = conv_param['stride']
        N, F, H_dout, W_dout = dout.shape
        F, C, HH, WW = w.shape
        db = torch.zeros_like(b,device="cuda")
        dw_bias = torch.zeros_like(w,device="cuda")
        A,B,X,Y = x.shape
        dx_bias = torch.zeros((A,B,X,Y), dtype=x.dtype, device= "cuda")

        dout_ptr = dout.flatten().contiguous().data_ptr()
        db_ptr = db.flatten().contiguous().data_ptr()
        x_ptr = x.flatten().contiguous().data_ptr()
        dw_ptr_bias = dw_bias.flatten().contiguous().data_ptr()
        dx_ptr_bias = dx_bias.flatten().contiguous().data_ptr()
        
        _curr_time = time.time()

        # Call CUDA kernels
        libconv.conv2d_backward_db(
            N, F, H_dout, W_dout, ctypes.cast(dout_ptr, ctypes.POINTER(ctypes.c_float)), ctypes.cast(db_ptr, ctypes.POINTER(ctypes.c_float)),
        )
        
        libconv.conv2d_backward_dw(
            N, C, H_dout, W_dout, ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float)), F, HH, WW,
            ctypes.cast(dout_ptr, ctypes.POINTER(ctypes.c_float)), ctypes.cast(dw_ptr_bias, ctypes.POINTER(ctypes.c_float)), H_dout, W_dout, stride, pad)
        
        reshaped_w = w.permute(1, 0, 2, 3)
        w_flipped = torch.flip(reshaped_w, dims=(2, 3))
        FF, CC, HH, WW = w_flipped.shape
        w_transpose = w_flipped.contiguous()
        w_ptr_transpose = w_transpose.flatten().data_ptr()
        N, C, H, W = x.shape
        libconv.conv2d(N, F, H, W, ctypes.cast(dout_ptr, ctypes.POINTER(ctypes.c_float)),
            FF, HH, WW, ctypes.cast(w_ptr_transpose, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(dx_ptr_bias, ctypes.POINTER(ctypes.c_float)), pad, stride)

        _time= (time.time() - _curr_time)  
        # print("Time taken by Backward Layer is: ",_time)

        dx_bias = dx_bias.reshape(A,B,X,Y)
        

        return dx_bias, dw_bias, db   


class Python_MaxPool(object):

    @staticmethod
    def forward(x, pool_param, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        # Extract pooling parameters
        stride = pool_param['stride']
        pool_width = pool_param['pool_width']
        pool_height = pool_param['pool_height']

        # Get input dimensions
        N, C, H, W = x.shape

        # Calculate output dimensions
        H_out = int(1 + (H - pool_height) / stride)
        W_out = int(1 + (W - pool_width) / stride)

        # Allocate memory for output and positions on GPU
    
        out = torch.zeros((N, C, H_out, W_out), dtype=x.dtype, device="cuda")
        positions = torch.zeros((N, C, H_out, W_out), dtype=torch.int32, device="cuda")
        # Ensure input tensor is on GPU and contiguous
        x_gpu = x.contiguous().cuda() if not x.is_cuda else x.contiguous()
        positions_gpu = positions.contiguous().cuda()if not positions.is_cuda else positions.contiguous()
        # Get pointers to the data
        x_ptr = x_gpu.flatten().data_ptr()
        out_ptr = out.flatten().data_ptr()
        pos_ptr = positions_gpu.flatten().data_ptr()
        _curr_time = time.time()
        # Launch the kernel
        libconv.max_pooling_forward(N, C, H, W, 
                                    ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float)), 
                                    ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_float)), 
                                    ctypes.cast(pos_ptr, ctypes.POINTER(ctypes.c_float)), 
                                    pool_height, pool_width, stride)

        # Ensure synchronization of CUDA operations
        
        out = out.reshape(N, C, H_out, W_out)
        cache = (x, positions_gpu, pool_param)
        _time= (time.time() - _curr_time)  
        # print("Time taken by Pooling Layer",layer_no,"is: ",_time)
                
        return out, cache

    @staticmethod
    def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        x, positions, pool_param = cache

        N, C, H, W = x.shape
        stride = pool_param['stride']
        pool_width = pool_param['pool_width']
        pool_height = pool_param['pool_height']

        # Create an output tensor dx
        dx = torch.zeros((N, C, H, W), dtype=x.dtype, device="cuda")

        # Convert tensors to contiguous if they are not already
        dout_ptr = dout.flatten().contiguous().data_ptr()
        positions_ptr = positions.flatten().contiguous().data_ptr()
        dx_ptr = dx.flatten().contiguous().data_ptr()
        
        # Call the CUDA function
        libconv.max_pooling_backward(N, C, H, W, ctypes.cast(dout_ptr, ctypes.POINTER(ctypes.c_float)),  ctypes.cast(dx_ptr, ctypes.POINTER(ctypes.c_float)),
                                      ctypes.cast(positions_ptr, ctypes.POINTER(ctypes.c_int32)),
                                    pool_height, pool_width, stride)

        dx = dx.reshape(N, C, H, W)

        return dx


class Python_BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, running_mean, running_var, Mode):

        # mode = 'train'
        bn_params = {'running_mean': running_mean, 'running_var': running_var}
        eps = bn_params.get('eps', 1e-5)
        momentum = bn_params.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_params.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
        running_var = bn_params.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))

        out, cache = None, None
        if Mode == "Training":

            # step1: calculate mean
            mu = 1. / N * torch.sum(x, axis=0)
            running_mean = momentum * running_mean + (1 - momentum) * mu

            # step2: subtract mean vector of every trainings example
            xmu = x - mu

            # step3: following the lower branch - calculation denominator
            sq = xmu ** 2

            # step4: calculate variance
            var = 1. / N * torch.sum(sq, axis=0)
            running_var = momentum * running_var + (1 - momentum) * var
            # step5: add eps for numerical stability, then sqrt
            sqrtvar = torch.sqrt(var + eps)

            # step6: invert sqrtwar
            ivar = 1. / sqrtvar

            # step7: execute normalization
            xhat = xmu * ivar

            # step8: Nor the two transformation steps
            # print(gamma)

            gammax = gamma * xhat

            # step9
            out = gammax + beta

            cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)

        elif Mode == "Inference":

            normolized = ((x - running_mean) / (running_var + eps) ** (1 / 2))
            out = normolized * gamma + beta

        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % Mode)

        # Store the updated running means back into bn_params
        bn_params['running_mean'] = running_mean.detach()
        bn_params['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        dx, dgamma, dbeta = None, None, None

        xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache

        N, D = dout.shape

        # step9
        dbeta = torch.sum(dout, axis=0)
        dgammax = dout  # not necessary, but more understandable

        # step8
        dgamma = torch.sum(dgammax * xhat, axis=0)
        dxhat = dgammax * gamma

        # step7
        divar = torch.sum(dxhat * xmu, axis=0)
        dxmu1 = dxhat * ivar

        # step6
        dsqrtvar = -1. / (sqrtvar ** 2) * divar

        # step5
        dvar = 0.5 * 1. / torch.sqrt(var + eps) * dsqrtvar

        # step4
        dsq = 1. / N * torch.ones((N, D), device=dout.device) * dvar

        # step3
        dxmu2 = 2 * xmu * dsq

        # step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * torch.sum(dxmu1 + dxmu2, axis=0)

        # step1
        dx2 = 1. / N * torch.ones((N, D), device=dout.device) * dmu

        # step0
        dx = dx1 + dx2

        return dx, dgamma, dbeta

    @staticmethod
    # def backward_alt(dout, cache):
    def backward_alt(dout, cache):
        dx, dgamma, dbeta = None, None, None

        xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache
        N, D = dout.shape
        # get the dimensions of the input/output
        dbeta = torch.sum(dout, dim=0)
        dgamma = torch.sum(xhat * dout, dim=0)
        dx = (gamma * ivar / N) * (N * dout - xhat * dgamma - dbeta)

        return dx, dgamma, dbeta


class Python_ReLU(object):

    @staticmethod
    def forward(x, alpha=0.1):

        out = None
        out = x.clone()
        out[out < 0] = out[out < 0] * alpha
        cache = x

        return out, cache

    @staticmethod
    def backward(dout, cache, alpha=0.1):
        dx, x = None, cache

        dl = torch.ones_like(x)
        dl[x < 0] = alpha
        dx = dout * dl

        return dx


class Python_Conv_ReLU(object):

    @staticmethod
    def forward(x, w, conv_param):
        a, conv_cache = Python_Conv.forward(x,w,conv_param)
        out, relu_cache = Python_ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, relu_cache = cache
        da = Python_ReLU.backward(dout,relu_cache)
        dx, dw = Python_Conv.backward(da, conv_cache)
        return dx, dw


class Python_Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, conv_param, pool_param):
        a, conv_cache = Python_Conv.forward(x, w, conv_param)
        s, relu_cache = Python_ReLU.forward(a)
        out, pool_cache = Python_MaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, relu_cache, pool_cache = cache
        ds = Python_MaxPool.backward(dout, pool_cache)
        da = Python_ReLU.backward(ds, relu_cache)
        dx, dw = Python_Conv.backward(da, conv_cache)
        return dx, dw


class Python_Conv_Pool(object):

    @staticmethod
    def forward(x, w, conv_param, pool_param):
        a, conv_cache = Python_Conv.forward(x, w, conv_param)
        out, pool_cache = Python_MaxPool.forward(a, pool_param)
        cache = (conv_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, pool_cache = cache
        ds = Python_MaxPool.backward(dout, pool_cache)
        dx, dw = Python_Conv.backward(ds, conv_cache)
        return dx, dw


class Python_Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, gamma, beta, conv_param, running_mean, running_var, mean, var):
        a, conv_cache = Python_Conv.forward(x, w, conv_param)
        an, bn_cache = Python_SpatialBatchNorm.forward(a, gamma, beta, running_mean, running_var, mean, var)
        out, relu_cache = Python_ReLU.forward(an)

        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = Python_ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = Python_SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw = Python_Conv.backward(da, conv_cache)
        return dx, dw, dgamma, dbeta


class Python_Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, gamma, beta, conv_param, running_mean, running_var, mean, var, pool_param):
        a, conv_cache = Python_Conv.forward(x, w, conv_param,)
        an, bn_cache = Python_SpatialBatchNorm.forward(a, gamma, beta, running_mean, running_var, mean, var)
        s, relu_cache = Python_ReLU.forward(an)
        out, pool_cache = Python_MaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = Python_MaxPool.backward(dout, pool_cache)
        dan = Python_ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = Python_SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw = Python_Conv.backward(da, conv_cache)

        return dx, dw, dgamma, dbeta
    
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

class Cal_mean_var_junaid(object):

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
        scale = 1 / ((avg_max - avg_min) * scale_fix + eps)  

        avg = avg.view(1, -1, 1, 1)
        scale = scale.view(1, -1, 1, 1)

        cache = x
        return avg, scale


class Python_SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, running_mean, running_var, mean, var):
        out, cache = None, None
        gamma = gamma.to(x.device)
        beta = beta.to(x.device)
        running_mean = running_mean.to(x.device)
        running_var = running_var.to(x.device)
        mean = mean.to(x.device)
        var = var.to(x.device)
        eps = 1e-5
        D = gamma.shape[0]
        num_chunks = 8
        running_mean = running_mean
        running_var = running_var
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
        output = (x - mean) * var
        output = output * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)

        running_mean = running_mean * momentum + (1 - momentum) * avg
        running_var = running_var * momentum + (1 - momentum) * scale
        cache = (x, gamma, beta, output, var, scale_fix, mean, avg_max, avg_min, eps, num_chunks, max_index, min_index)
        
        return output, cache
    
    @staticmethod
    def backward(grad_output, cache):
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
        dL_dgamma = (grad_output * output).sum(dim=(0, 2, 3), keepdim=True) # TO DO - Is it really required to keep dim
        dL_dbeta = grad_output.sum(dim=(0, 2, 3), keepdim=True)
        dL_davg = grad_output.sum(dim=(0, 2, 3), keepdim=True)

        # Average per channel
        avg_pc = (dL_dxi_hat * -1.0).sum(dim=(0, 2, 3), keepdim=True) / (B * H * W)
        dL_dxi_ = avg_pc + dL_dxi_hat
        
        # Backward coefficient
        backward_const = scale
        
        # Final output calculation
        dL_dxi = dL_dxi_ * backward_const

        return dL_dxi, dL_dgamma, dL_dbeta     
