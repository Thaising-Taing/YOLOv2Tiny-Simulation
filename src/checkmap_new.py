import os
import sys
import torch
from torch import optim
import pdb
from copy import deepcopy
import tkinter
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
from pypcie import Device
from ast import literal_eval
import shutil
import matplotlib
import matplotlib.pyplot as plt
import subprocess
import tqdm
import pickle
import numpy as np
import argparse
import warnings
from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader
sys.path.append("../")
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),"Dataset"))
sys.path.append(os.path.join(os.getcwd(),"src"))
sys.path.append(os.path.join(os.getcwd(),"src/GiTae"))
sys.path.append(os.path.join(os.getcwd(),"src/config"))
sys.path.append(os.path.join(os.getcwd(),"src/Main_Processing_Scratch"))
sys.path.append(os.path.join(os.getcwd(),"src/Pre_Processing_Scratch"))
sys.path.append(os.path.join(os.getcwd(),"src/Post_Processing_Scratch"))
sys.path.append(os.path.join(os.getcwd(),"src/Weight_Update_Algorithm"))
sys.path.append(os.path.join(os.getcwd(),"src/Wathna"))
sys.path.append("/home/msis/Desktop/pcie_python/GUI")
from Dataset.factory import *
from Dataset.roidb import *
from Dataset.imdb import *
from Dataset.pascal_voc import *
from util.visualize import *
from Dataset.yolo_eval import *
from src.batchnorm_pytorch import *
from src.Wathna_pytorch import *
from src.Thaising_PyTorch import *
from src.Thaising_PyTorch_BatchNorm import *
from src.GiTae import *

from tqdm import tqdm

# pytorch_model = Pytorch_bn("none")
# with open('./Dataset/Dataset/pretrained/epoch_548.pkl', 'rb') as f:
#     x = pickle.load(f)
# model = x['model']
# checkpoint = torch.load('./Dataset/Dataset/pretrained/yolov2_epoch_548.pth')

# for param, val in checkpoint['model'].items():
#     for param_mod, val_mod in pytorch_model.modtorch_model.params.items():
#         if (param == param_mod):
#             pytorch_model.modtorch_model.params[param] = val

# pytorch_model.get_weights()

# with open('./random_search_weights/scratch1/epoch_19.pkl', 'rb') as f:
#     x = pickle.load(f)
# model = x['model']

# with open('./weight_test/scratch_zero_grad3_590_big_lr/epoch_299.pkl', 'rb') as f:
#     x = pickle.load(f)
# model = x['model']
# with open('weight_test/scratch_zero_momentum/epoch_45.pkl', 'rb') as f:
#     x = pickle.load(f)


def parse_args():

    parser = argparse.ArgumentParser('Yolo v2')
    parser.add_argument('--dataset', dest='dataset',
                        default='voc07test', type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output', type=str)
    parser.add_argument('--model_name', dest='model_name',
                        default='yolov2_epoch_160.pth', type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of workers to load training data',
                        default=1, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        default=8, type=int)
    parser.add_argument('--cuda', dest='use_cuda',
                        default=False, type=bool)
    parser.add_argument('--vis', dest='vis',
                        default=False, type=bool)

    args = parser.parse_args()
    return args


def prepare_im_data(img):
    """
    Prepare image data that will be feed to network.

    Arguments:
    img -- PIL.Image object

    Returns:
    im_data -- tensor of shape (3, H, W).
    im_info -- dictionary {height, width}

    """

    im_info = dict()
    im_info['width'], im_info['height'] = img.size

    # resize the image
    H, W = cfg.input_size
    im_data = img.resize((H, W))

    # to torch tensor
    im_data = torch.from_numpy(np.array(im_data)).float() / 255

    im_data = im_data.permute(2, 0, 1).unsqueeze(0)

    return im_data, im_info


def Forward_pred(out):
    """
    Evaluate loss and gradient for the deep convolutional network.
    Input / output: Same API as ThreeLayerConvNet.
    """
    # print('Calculating the loss and its gradients for pytorch model.')

    scores = out
    bsize, _, h, w = out.shape
    out = out.permute(0, 2, 3, 1).contiguous().view(bsize, 13 * 13 * 5, 5 + 20)
    # Calculate losses based on loss functions(box loss, Intersection over Union(IoU) loss, class loss)
    xy_pred = torch.sigmoid(out[:, :, 0:2]) #
    conf_pred = torch.sigmoid(out[:, :, 4:5]) # 
    hw_pred = torch.exp(out[:, :, 2:4])
    class_score = out[:, :, 5:]
    class_pred = F.softmax(class_score, dim=-1)
    delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

    # dout = open("./Pytorch_Backward_loss_gradients.pickle", "rb")
    # dout = pickle.load(dout)
    # print('\n\n',dout.dtype, dout[dout!=0])
    return delta_pred, conf_pred, class_pred
    

def check(weights=[], pth='', args=[], mode = "PytorchSim", _data=[]):


    if args==[]:
        args = parse_args()
    args.conf_thresh = 0.005
    args.nms_thresh = 0.45
    if args.vis:
        args.conf_thresh = 0.5

    # prepare dataset
    args.dataset = 'voc07test'
    args.imdbval_name = 'voc_2007_test'
    # args.dataset = 'random-64'
    # args.imdbval_name = 'voc_2007_trainval-random-64'
    val_imdb = get_imdb(args.imdbval_name)
    val_dataset = RoiDataset(val_imdb, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    dataset_size = len(val_imdb.image_index)
    all_boxes = [[[] for _ in range(dataset_size)] for _ in range(val_imdb.num_classes)]
    det_file = os.path.join(args.output_dir, 'detections.pkl')
    img_id = -1



    print(f"\n\nVal Model: ", mode)

    model = _data

    if not weights==[]:      
        model.load_weights(weights)
        
    if   mode == "Pytorch_BN"    or mode == "Pytorch_LN"    : model = model.modtorch_model
    elif mode == "Python_BN"     or mode == "Python"        : model = model.python_model
    elif mode == "PytorchSim_BN" or mode == "PytorchSim_LN" : model = model
    elif mode == "FPGA"                                     : model.Forward = model.Forward_Inference
    elif mode == "RFFP_CUDA"                                : model = model.python_model
            
    if weights==[]:      
        checkpoint = torch.load(pth)
        # checkpoint = torch.load('./Dataset/Dataset/pretrained/yolov2_epoch_548.pth') 
        for param, val in checkpoint['model'].items():
            for param_mod, val_mod in model.params.items():
                if (param == param_mod):
                    model.params[param] = val
        
    with torch.no_grad():
        for batch, (im_data, im_infos) in tqdm(enumerate(val_dataloader), desc='Checking mAP', total=len(val_dataloader)):
            im_data_variable = Variable(im_data.cuda())

            out, _, _ = model.Forward(im_data_variable)
            yolo_outputs = Forward_pred(out)
            
            for i in range(im_data.size(0)):
                img_id += 1
                output = [item[i].data for item in yolo_outputs]
                im_info = {'width': im_infos[i][0], 'height': im_infos[i][1]}
                detections = yolo_eval(output, im_info, conf_threshold=args.conf_thresh,
                                    nms_threshold=args.nms_thresh)
                # print('im detect [{}/{}]'.format(img_id+1, len(val_dataset)))
                if len(detections) > 0:
                    for cls in range(val_imdb.num_classes):
                        inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                        if inds.numel() > 0:
                            cls_det = torch.zeros((inds.numel(), 5))
                            cls_det[:, :4] = detections[inds, :4]
                            cls_det[:, 4] = detections[inds, 4] * detections[inds, 5]
                            all_boxes[cls][img_id] = cls_det.cpu().numpy()

                if args.vis:
                    img = Image.open(val_imdb.image_path_at(img_id))
                    if len(detections) == 0:
                        continue
                    det_boxes = detections[:, :5].cpu().numpy()
                    det_classes = detections[:, -1].long().cpu().numpy()
                    im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=val_imdb.classes)
                    plt.figure()
                    plt.imshow(im2show)
                    plt.show()

    # with open(det_file, 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    mAP = val_imdb.evaluate_detections(all_boxes, output_dir=args.output_dir)
    return mAP

    

if __name__ == '__main__':
    check()















