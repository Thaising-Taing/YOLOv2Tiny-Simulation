# --------------------------------------------------------
# Pytorch Yolov2
# Licensed under The MIT License [see LICENSE for details]
# Written by Jingru Tan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import math

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
from pathlib import Path
# from config import config as cfg
from Post_Processing_Scratch.Loss import build_target, yolo_loss


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
        # mantissa = int(value[1 + Exponent_Bit:], 2) * 2 ** (-Mantissa_Bit)
        mantissa = int(value[1 + Exponent_Bit:], 2) * math.pow(2, (-Mantissa_Bit))
    Floating_Decimal = (math.pow(-1, sign)) * (math.pow(2, exponent)) * mantissa
    return Floating_Decimal

def Truncating_Rounding(Truncated_Hexadecimal):
    # Consider only the Length of Truncated_Hexadecimal only in [0:5]
    if len(Truncated_Hexadecimal) >= 5:
        # If this Truncated_Hexadecimal[4] >= 5 => Rounding Up the First 16 Bits
        if int(Truncated_Hexadecimal[4], 16) >= 8:
            Rounding_Hexadecimal = hex(int(Truncated_Hexadecimal[:4], 16) + 1)[2:]
        else:
            Rounding_Hexadecimal = Truncated_Hexadecimal[:4]
    else:
        Rounding_Hexadecimal = Truncated_Hexadecimal

    Rounding_Hexadecimal_Capitalized = Rounding_Hexadecimal.upper()

    return Rounding_Hexadecimal_Capitalized

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

def save_file(fname, data, module=[], layer_no=[], save_txt=False, save_hex=False, phase=[]):
  
  if save_txt or save_hex:
    if type(data) is dict:
      for _key in data.keys():
        _fname = fname+f'_{_key}'
        save_file(_fname,data[_key])
    
    else:
      if module==[] and layer_no==[]: 
        Out_Path = f'Outputs/{os.path.split(fname)[0]}'
        fname = os.path.split(fname)[1]
      else:
        Out_Path = f'Outputs/By_Layer/'
        if layer_no!=[]: Out_Path+= f'Layer{layer_no}/'
        if module!=[]: Out_Path+= f'{module}/'
        if phase!=[]: Out_Path+= f'{phase}/'
        fname=fname
        
      if save_txt: filename = os.path.join(Out_Path, fname+'.txt')
      if save_hex: hexname  = os.path.join(Out_Path, fname+'_hex.txt')
      
      Path(Out_Path).mkdir(parents=True, exist_ok=True)
      
      if torch.is_tensor(data):
        try: data = data.detach()
        except: pass
        data = data.numpy()
      
      if save_txt: outfile = open(filename  , mode='w')
      if save_txt: outfile.write(f'{data.shape}\n')
      
      if save_hex: hexfile = open(hexname, mode='w')
      if save_hex: hexfile.write(f'{data.shape}\n')
      
      if len(data.shape)==0:
        if save_txt: outfile.write(f'{data}\n')
        if save_hex: hexfile.write(f'{data}\n')
        pass
      elif len(data.shape)==1:
        for x in data:
          if save_txt: outfile.write(f'{x}\n')
          if save_hex: hexfile.write(f'{convert_to_hex(x)}\n')
          pass
      else:
        w,x,y,z = data.shape
        for _i in range(w):
          for _j in range(x):
            for _k in range(y):
              for _l in range(z):
                _value = data[_i,_j,_k,_l]
                if save_txt: outfile.write(f'{_value}\n')
                if save_hex: hexfile.write(f'{convert_to_hex(_value)}\n')
                pass
                
      if save_hex: hexfile.close()  
      if save_txt: outfile.close()  
      
      if save_txt: print(f'\t\t--> Saved {filename}')
      if save_hex: print(f'\t\t--> Saved {hexname}')
  # else:
      # print(f'\n\t\t--> Saved {filename}')

def save_cache(fname,data):
  
  if type(data) is dict:
    for _key in data.keys():
      _fname = fname+f'_{_key}'
      save_file(_fname,data[_key])
  
  else:
    Path(os.path.split(fname)[0]).mkdir(parents=True, exist_ok=True)
    fname = fname+'.txt'
    
    if torch.is_tensor(data):
      try: data = data.detach()
      except: pass
      data = data.numpy()
    
    outfile = open(fname, mode='w')
    outfile.write(f'{data.shape}\n')
    
    if len(data.shape)==0:
      outfile.write(f'{data}\n')
    elif len(data.shape)==1:
      for x in data:
        outfile.write(f'{x}\n')
    else:
      w,x,y,z = data.shape
      for _i in range(w):
        for _j in range(x):
          for _k in range(y):
            for _l in range(z):
              outfile.write(f'{data[_i,_j,_k,_l]}\n')
    outfile.close()  
    
    print(f'\n\t\t--> Saved {fname}')

class RangeBN(nn.Module):
    def __init__(self, num_features, momentum=0.1, affine=True, num_chunks=8, eps=1e-5):
        super(RangeBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.momentum = momentum
        if affine:
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight = nn.Parameter(torch.Tensor(num_features))
        self.eps = eps
        self.num_chunks = num_chunks
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            self.weight.data.uniform_()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, calculated_mean, calculated_var):
        input_ = x
        gamma_ = self.weight
        #if self.training:
        B, C, H, W = input_.shape
        y = input_.transpose(0, 1).contiguous()  # C x B x H x W
        y = y.view(C, self.num_chunks, B * H * W // self.num_chunks)
        mean_max = y.max(-1)[0].mean(-1)  # C
        mean_min = y.min(-1)[0].mean(-1)  # C
        mean = y.view(C, -1).mean(-1)  # C
        #scale_fix = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) **
        #                            0.5) / ((2 * math.log(y.size(-1))) ** 0.5)
        scale_fix = 1 / ((2 * math.log(y.size(-1))) ** 0.5)
        scale = 1 / ((mean_max - mean_min) * scale_fix + self.eps)
        #print('scale', scale)
        self.running_mean.detach().mul_(self.momentum).add_(
            mean * (1 - self.momentum))

        self.running_var.detach().mul_(self.momentum).add_(
            scale * (1 - self.momentum))
        """else:
            mean = self.running_mean
            scale = self.running_var"""
        out = (x - calculated_mean) * calculated_var
        out = out * gamma_.view(1, gamma_.size(0), 1, 1) + self.bias.view(1, self.bias.size(0), 1, 1)

        return out

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

class  Yolov2(nn.Module):

    num_classes = 20
    num_anchors = 5

    def __init__(self, classes=None, weights_file=False):
        super(Yolov2, self).__init__()
        if classes:
            self.num_classes = len(classes)
            
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.slowpool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = RangeBN(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = RangeBN(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = RangeBN(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = RangeBN(128)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = RangeBN(256)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = RangeBN(512)

        self.conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = RangeBN(1024)

        self.conv8 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = RangeBN(1024)

        self.conv9 = nn.Sequential(nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1))

    def forward(self, x, gt_boxes=None, gt_classes=None, num_boxes=None, training=False):
        """
        x: Variable
        gt_boxes, gt_classes, num_boxes: Tensor
        """
        temp_x = self.maxpool(self.conv1(x))
        cal_mean, cal_var = Cal_mean_var.forward(temp_x)
        
        
        x = self.maxpool(self.lrelu(self.bn1(self.conv1(x), cal_mean, cal_var)))
        
        temp_x = self.conv2(x)
        cal_mean, cal_var = Cal_mean_var.forward(temp_x)
        
        x = self.maxpool(self.lrelu(self.bn2(self.conv2(x), cal_mean, cal_var)))
        
        temp_x = self.conv3(x)
        cal_mean, cal_var = Cal_mean_var.forward(temp_x)

        x = self.maxpool(self.lrelu(self.bn3(self.conv3(x), cal_mean, cal_var)))
        
        temp_x = self.conv4(x)
        cal_mean, cal_var = Cal_mean_var.forward(temp_x)

        x = self.maxpool(self.lrelu(self.bn4(self.conv4(x), cal_mean, cal_var)))
        
        temp_x = self.conv5(x)
        cal_mean, cal_var = Cal_mean_var.forward(temp_x)

       
        x = self.maxpool(self.lrelu(self.bn5(self.conv5(x), cal_mean, cal_var)))
        
        temp_x = self.conv6(x)
        cal_mean, cal_var = Cal_mean_var.forward(temp_x)
        
        x = self.lrelu(self.bn6(self.conv6(x), cal_mean, cal_var))
        
        temp_x = self.conv7(x)
        cal_mean, cal_var = Cal_mean_var.forward(temp_x)
        
 
        x = self.lrelu(self.bn7(self.conv7(x), cal_mean, cal_var))
        
        temp_x = self.conv8(x)
        cal_mean, cal_var = Cal_mean_var.forward(temp_x)
 
        x = self.lrelu(self.bn8(self.conv8(x), cal_mean, cal_var))

        out = self.conv9(x)
    


        # out -- tensor of shape (B, num_anchors * (5 + num_classes), H, W)
        bsize, _, h, w = out.size()

        # 5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
        # reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)
    
        # 5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
        # reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)
        out = out.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * self.num_anchors, 5 + self.num_classes)

        # activate the output tensor
        # `sigmoid` for t_x, t_y, t_c; `exp` for t_h, t_w;
        # `softmax` for (class1_score, class2_score, ...)

        xy_pred = torch.sigmoid(out[:, :, 0:2])
        conf_pred = torch.sigmoid(out[:, :, 4:5])
        hw_pred = torch.exp(out[:, :, 2:4])
        class_score = out[:, :, 5:]
        class_pred = F.softmax(class_score, dim=-1)
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

        if training:
            output_variable = (delta_pred, conf_pred, class_score)
            output_data = [v.data for v in output_variable]
            gt_data = (gt_boxes, gt_classes, num_boxes)
            target_data = build_target(output_data, gt_data, h, w)

            target_variable = [Variable(v) for v in target_data]
            box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)

            return box_loss, iou_loss, class_loss

        return delta_pred, conf_pred, class_pred

if __name__ == '__main__':
    model = Yolov2()
    print(model)
    im = np.random.randn(1, 3, 416, 416)
    im_variable = Variable(torch.from_numpy(im)).float()
    out = model(im_variable)
    delta_pred, conf_pred, class_pred = out
    print('delta_pred size:', delta_pred.size())
    print('conf_pred size:', conf_pred.size())
    print('class_pred size:', class_pred.size())