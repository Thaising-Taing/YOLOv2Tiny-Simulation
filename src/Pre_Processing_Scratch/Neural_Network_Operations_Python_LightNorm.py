from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import warnings
import torch
from pathlib import Path
import math

warnings.simplefilter("ignore", UserWarning)

# Python_Convolution without Bias
class Python_Conv(object):

    @staticmethod
    def forward(x, w, conv_param):

        out = None

        pad = conv_param['pad']
        stride = conv_param['stride']
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        H_out = int(1 + (H + 2 * pad - HH) / stride)
        W_out = int(1 + (W + 2 * pad - WW) / stride)
        x = torch.nn.functional.pad(x, (pad, pad, pad, pad))

        out = torch.zeros((N, F, H_out, W_out), dtype=x.dtype, device=x.device)

        for n in range(N):
            for f in range(F):
                for height in range(H_out):
                    for width in range(W_out):
                        out[n, f, height, width] = (
                                x[n, :, height * stride:height * stride + HH, width * stride:width * stride + WW] *
                                w[f]).sum()

        cache = (x, w, conv_param)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        x, w, conv_param = cache

        dx, dw = None, None

        pad = conv_param['pad']
        stride = conv_param['stride']
        N, F, H_dout, W_dout = dout.shape
        F, C, HH, WW = w.shape
        dw = torch.zeros_like(w)
        dx = torch.zeros_like(x)
        for n in range(N):
            for f in range(F):
                for height in range(H_dout):
                    for width in range(W_dout):
                        dw[f] += x[n, :, height * stride:height * stride + HH, width * stride:width * stride + WW] * \
                                 dout[n, f, height, width]
                        dx[n, :, height * stride:height * stride + HH, width * stride:width * stride + WW] += w[f] * \
                                                                                                              dout[
                                                                                                                  n, f, height, width]

        dx = dx[:, :, 1:-1, 1:-1]  # delete padded "pixels"

        return dx, dw


# Python_Convolution with Bias
class Python_ConvB(object):

    @staticmethod
    def forward(x, w, b, conv_param):

        out = None

        pad = conv_param['pad']
        stride = conv_param['stride']
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        H_out = int(1 + (H + 2 * pad - HH) / stride)
        W_out = int(1 + (W + 2 * pad - WW) / stride)
        x = torch.nn.functional.pad(x, (pad, pad, pad, pad))

        out = torch.zeros((N, F, H_out, W_out), dtype=x.dtype, device=x.device)

        for n in range(N):
            for f in range(F):
                for height in range(H_out):
                    for width in range(W_out):
                        out[n, f, height, width] = (x[n, :, height * stride:height * stride + HH,
                                                    width * stride:width * stride + WW] * w[f]).sum() + b[f]

        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        x, w, b, conv_param = cache

        dx, dw, db = None, None, None

        pad = conv_param['pad']
        stride = conv_param['stride']
        N, F, H_dout, W_dout = dout.shape
        F, C, HH, WW = w.shape
        db = torch.zeros_like(b)
        dw = torch.zeros_like(w)
        dx = torch.zeros_like(x)
        for n in range(N):
            for f in range(F):
                for height in range(H_dout):
                    for width in range(W_dout):
                        db[f] += dout[n, f, height, width]
                        dw[f] += x[n, :, height * stride:height * stride + HH, width * stride:width * stride + WW] * \
                                 dout[n, f, height, width]
                        dx[n, :, height * stride:height * stride + HH, width * stride:width * stride + WW] += w[f] * \
                                                                                                              dout[
                                                                                                                  n, f, height, width]
        if pad != 0:
            dx = dx[:, :, 1:-1, 1:-1]  # delete padded "pixels"

        return dx, dw, db


class Python_MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        out = None
        stride = pool_param['stride']
        pool_width = pool_param['pool_width']
        pool_height = pool_param['pool_height']
        N, C, H, W = x.shape
        H_out = int(1 + (H - pool_height) / stride)
        W_out = int(1 + (W - pool_width) / stride)
        out = torch.zeros((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
        
        for n in range(N):
            for height in range(H_out):
                for width in range(W_out):
                    val, index = x[n, :, height * stride:height * stride + pool_height,
                                 width * stride:width * stride + pool_width].reshape(C, -1).max(dim=1)
                    out[n, :, height, width] = val

        positions = []
        _out = torch.zeros((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        _val, _idx = x[n, c, h * stride:h * stride + pool_height,
                                     w * stride:w * stride + pool_width].reshape(-1).max(dim=0)
                        input_tensor = x[n, c, h * stride:h * stride + pool_height,
                                       w * stride:w * stride + pool_width].reshape(-1)
                        max_index = torch.argmax(input_tensor)
                        max_value = input_tensor[max_index]
                        all_max_indices = torch.nonzero(input_tensor == max_value).flatten()
                        last_index_of_highest_value = all_max_indices[-1].item()
                        positions.append(last_index_of_highest_value)
                        _out[n, c, h, w] = _val

        cache = (x, pool_param)
        positions = torch.tensor(positions)

        return out, cache

    @staticmethod
    def backward(dout, cache):

        x, pool_param = cache

        dx = None

        N, C, H, W = x.shape
        stride = pool_param['stride']
        pool_width = pool_param['pool_width']
        pool_height = pool_param['pool_height']

        H_out = int(1 + (H - pool_height) / stride)
        W_out = int(1 + (W - pool_width) / stride)
        dx = torch.zeros_like(x)
        
        backward_positions = []
        
        
        for n in range(N):
            for c in range(C):
                temp_positions = []
                for height in range(H_out):
                    for width in range(W_out):
                        local_x = x[n, c, height * stride:height * stride + pool_height,
                                width * stride:width * stride + pool_width]
                        
                        shape_local_x = local_x.shape
                        
                        
                        input_tensor = local_x.reshape(-1)
                        
                        # print("input_tensor",input_tensor)
                        
                        # print("input_tensor", input_tensor.shape, input_tensor)
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
        scale = 1 / ((avg_max - avg_min) * scale_fix + eps)  

        avg = avg.view(1, -1, 1, 1)
        scale = scale.view(1, -1, 1, 1)

        cache = x
        return avg, scale


class Python_SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, running_mean, running_var, mean, var):
        out, cache = None, None
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
