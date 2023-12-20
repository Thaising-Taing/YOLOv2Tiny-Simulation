import torch
import math
import os
from pathlib import Path


class Torch_Conv(object):

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
        dx, dw, db = None, None, None

        x, w, conv_param = cache
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
        print(dx.shape)

        return dx, dw


class Torch_ConvB(object):

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
        dx, dw, db = None, None, None

        x, w, b, conv_param = cache
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

        dx = dx[:, :, 1:-1, 1:-1]  # delete padded "pixels"

        return dx, dw, db


class Torch_MaxPool(object):

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
                    val, _ = x[n, :, height * stride:height * stride + pool_height,
                             width * stride:width * stride + pool_width].reshape(C, -1).max(dim=1)
                    out[n, :, height, width] = val

        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        dx = None

        x, pool_param = cache
        N, C, H, W = x.shape
        stride = pool_param['stride']
        pool_width = pool_param['pool_width']
        pool_height = pool_param['pool_height']

        H_out = int(1 + (H - pool_height) / stride)
        W_out = int(1 + (W - pool_width) / stride)
        dx = torch.zeros_like(x)
        for n in range(N):
            for height in range(H_out):
                for width in range(W_out):
                    local_x = x[n, :, height * stride:height * stride + pool_height,
                              width * stride:width * stride + pool_width]
                    shape_local_x = local_x.shape
                    reshaped_local_x = local_x.reshape(C, -1)
                    local_dw = torch.zeros_like(reshaped_local_x)
                    values, indicies = reshaped_local_x.max(-1)
                    local_dw[range(C), indicies] = dout[n, :, height, width]
                    dx[n, :, height * stride:height * stride + pool_height,
                    width * stride:width * stride + pool_width] = local_dw.reshape(shape_local_x)

        return dx


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
    
    @staticmethod
    def backward(dout, cache):
        
        x = cache
        B, C, H, W = x.shape
        dL_davg = (dout).sum(dim=(0, 2, 3), keepdim=True)
        avg_pc = dL_davg / (B * H * W)
        
        
        return avg_pc

class Torch_SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, running_mean, running_var, mean, var, Mode):  
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
        
        cache = (x, gamma, beta, output, var, scale, mean, avg_max, avg_min, eps, num_chunks, max_index, min_index)
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

class Torch_FastConv(object):

    @staticmethod
    def forward(x, w, conv_param):
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


class Torch_FastConvWB(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        x, _, _, _, tx, out, layer = cache
        out.backward(dout)
        dx = tx.grad.detach()
        dw = layer.weight.grad.detach()
        db = layer.bias.grad.detach()
        layer.weight.grad = layer.bias.grad = None
        return dx, dw, db


class Torch_FastMaxPool(object):
    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width), stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        global tx
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Torch_ReLU(object):

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


class Torch_Conv_ReLU(object):

    @staticmethod
    def forward(x, w, conv_param):
        a, conv_cache = Torch_FastConv.forward(x, w, conv_param)
        out, relu_cache = Torch_ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, relu_cache = cache
        da = Torch_ReLU.backward(dout, relu_cache)
        dx, dw = Torch_FastConv.backward(da, conv_cache)
        return dx, dw


class Torch_Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, conv_param, pool_param):
        a, conv_cache = Torch_FastConv.forward(x, w, conv_param)
        s, relu_cache = Torch_ReLU.forward(a)
        out, pool_cache = Torch_FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, relu_cache, pool_cache = cache
        ds = Torch_FastMaxPool.backward(dout, pool_cache)
        da = Torch_ReLU.backward(ds, relu_cache)
        dx, dw = Torch_FastConv.backward(da, conv_cache)
        return dx, dw


class Torch_Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, gamma, beta, conv_param, running_mean, running_var, mean, var, Mode):
        a, conv_cache = Torch_FastConv.forward(x, w, conv_param)
        an, bn_cache = Torch_SpatialBatchNorm.forward(a, gamma, beta, running_mean, running_var, mean, var, Mode)
        out, relu_cache = Torch_ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = Torch_ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = Torch_SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw = Torch_FastConv.backward(da, conv_cache)
        return dx, dw, dgamma, dbeta


class Torch_Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, gamma, beta, conv_param, running_mean, running_var, mean, var, Mode, pool_param):
        a, conv_cache = Torch_FastConv.forward(x, w, conv_param)
        an, bn_cache = Torch_SpatialBatchNorm.forward(a, gamma, beta, running_mean, running_var, mean, var, Mode)
        s, relu_cache = Torch_ReLU.forward(an)
        out, pool_cache = Torch_FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = Torch_FastMaxPool.backward(dout, pool_cache)
        dan = Torch_ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = Torch_SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw = Torch_FastConv.backward(da, conv_cache)
        return dx, dw, dgamma, dbeta

class Torch_Conv_Pool(object):

    @staticmethod
    def forward(x, w, conv_param, pool_param):
        a, conv_cache = Torch_FastConv.forward(x, w, conv_param)
        out, pool_cache = Torch_FastMaxPool.forward(a, pool_param)
        cache = (conv_cache, pool_cache)
        return out, cache
    
    @staticmethod
    def backward(dout, cache):
        conv_cache, pool_cache = cache
        ds = Torch_FastMaxPool.backward(dout,pool_cache)
        dx, dw = Torch_FastConv.backward(ds, conv_cache)
        return dx, dw