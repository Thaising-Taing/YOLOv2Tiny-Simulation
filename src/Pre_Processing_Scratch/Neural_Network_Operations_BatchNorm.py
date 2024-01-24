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
    
class Torch_BatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, running_mean, running_var, mode):
    eps = 1e-5
    momentum = 0.9
    
    N, D = x.shape
    running_mean = running_mean
    running_var = running_var

    out, cache = None, None
    if mode == "Training":
      mu = 1./N * torch.sum(x, axis = 0)
      running_mean = momentum * running_mean + (1 - momentum) * mu
      xmu = x - mu
      sq = xmu ** 2
      var = 1./N * torch.sum(sq, axis = 0)
      running_var = momentum * running_var + (1 - momentum) * var
      sqrtvar = torch.sqrt(var + eps)
      ivar = 1./sqrtvar
      xhat = xmu * ivar
      gammax = gamma * xhat
      out = gammax + beta
      cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)
      
    elif mode == "Test":
      normolized = ((x - running_mean)/(running_var+ eps)**(1/2))
      out = normolized * gamma + beta
    else:
      raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    return out, cache

  @staticmethod
  def backward(dout, cache):
    xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
    
    N,D = dout.shape
    dbeta = torch.sum(dout, axis=0)
    dgammax = dout #not necessary, but more understandable
    dgamma = torch.sum(dgammax*xhat, axis=0)
    dxhat = dgammax * gamma
    divar = torch.sum(dxhat*xmu, axis=0)
    dxmu1 = dxhat * ivar
    dsqrtvar = -1. /(sqrtvar**2) * divar
    dvar = 0.5 * 1. /torch.sqrt(var+eps) * dsqrtvar
    dsq = 1. /N * torch.ones((N,D),device = dout.device) * dvar
    dxmu2 = 2 * xmu * dsq
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * torch.sum(dxmu1+dxmu2, axis=0)
    dx2 = 1. /N * torch.ones((N,D),device = dout.device) * dmu
    dx = dx1 + dx2
    
    return dx, dgamma, dbeta

  @staticmethod
  def backward_alt(dout, cache):
    dx, dgamma, dbeta = None, None, None
    xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
    N,D = dout.shape
    # get the dimensions of the input/output
    dbeta = torch.sum(dout, dim=0)
    dgamma = torch.sum(xhat * dout, dim=0)
    dx = (gamma*ivar/N) * (N*dout - xhat*dgamma - dbeta)

    return dx, dgamma, dbeta

class Torch_SpatialBatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, running_mean, running_var, mode):
    out, cache = None, None
    N,C,H,W = x.shape
    pre_m = x.permute(1,0,2,3).reshape(C,-1).T
    pre_m_normolized, cache= Torch_BatchNorm.forward(pre_m, gamma, beta, running_mean, running_var, mode)
    out = pre_m_normolized.T.reshape(C, N, H, W).permute(1,0,2,3)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    dx, dgamma, dbeta = None, None, None
    N,C,H,W = dout.shape
    pre_m = dout.permute(1,0,2,3).reshape(C,-1).T
    dx, dgamma, dbeta = Torch_BatchNorm.backward_alt(pre_m, cache)
    dx =dx.T.reshape(C, N, H, W).permute(1,0,2,3)

    return dx, dgamma, dbeta

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
    def forward(x, w, gamma, beta, conv_param, running_mean, running_var, Mode):
        a, conv_cache = Torch_FastConv.forward(x, w, conv_param)
        an, bn_cache = Torch_SpatialBatchNorm.forward(a, gamma, beta, running_mean, running_var, Mode)
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
    def forward(x, w, gamma, beta, conv_param, running_mean, running_var, Mode, pool_param):
        a, conv_cache = Torch_FastConv.forward(x, w, conv_param)
        an, bn_cache = Torch_SpatialBatchNorm.forward(a, gamma, beta, running_mean, running_var, Mode)
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