import sys
sys.path.append("../")
import os
import argparse
import time
import numpy as np
import pickle
import torch
from torch.autograd import Variable
from PIL import Image
from Detection.yolov2_tiny import Yolov2
from Detection.yolo_eval import yolo_eval
from Dataset.factory import get_imdb
from Dataset.roidb import RoiDataset
from util.visualize import draw_detection_boxes
from Dataset.roidb import RoiDataset, detection_collate
import matplotlib.pyplot as plt
from util.network import WeightLoader
from torch.utils.data import DataLoader
from my_config import config as cfg
from tqdm import tqdm
from pathlib import Path


def parse_args():

    parser = argparse.ArgumentParser('Yolo v2')
    parser.add_argument('--dataset', dest='dataset',
                        default='voc07test', type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        default='Output', type=str)
    # parser.add_argument('--model_name', dest='model_name',
    #                     default='yolov2_epoch_160', type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of workers to load training data',
                        default=1, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        default=2, type=int)
    parser.add_argument('--cuda', dest='use_cuda',
                        default=True, type=bool)
    parser.add_argument('--vis', dest='vis',
                        default=True, type=bool)

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


def Detection(yolo_outputs, epoch):
    conf_thresh = 0.5
    nms_thresh = 0.5
    batch_size = 2
    output_dir = "Output"
    dataset = "voc07test"
    vis = True

    # prepare dataset

    if dataset == 'voc07trainval':
        imdbval_name = 'voc_2007_trainval'

    elif dataset == 'voc07test':
        imdbval_name = 'voc_2007_test'

    else:
        raise NotImplementedError

    val_imdb = get_imdb(imdbval_name)

    val_dataset = RoiDataset(val_imdb, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Small Dataset
    small_val_dataset = torch.utils.data.Subset(val_dataset, range(0, 8))
    print("Sub Training Dataset: " + str(len(small_val_dataset)))
    small_val_dataloader = DataLoader(small_val_dataset, batch_size=batch_size, shuffle=False)

    dataset_size = len(val_imdb.image_index)

    all_boxes = [[[] for _ in range(dataset_size)] for _ in range(val_imdb.num_classes)]

    det_file = os.path.join(output_dir, 'detections.pkl')

    img_id = -1

    with torch.no_grad():
        for batch, (im_data, im_infos) in enumerate(small_val_dataloader):
            for i in range(im_data.size(0)):
                img_id += 1
                output = [item[i].data for item in yolo_outputs]
                im_info = {'width': im_infos[i][0], 'height': im_infos[i][1]}
                detections = yolo_eval(output, im_info, conf_threshold=conf_thresh, nms_threshold=nms_thresh)

                if len(detections) > 0:
                    for cls in range(val_imdb.num_classes):
                        inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                        if inds.numel() > 0:
                            cls_det = torch.zeros((inds.numel(), 5))
                            cls_det[:, :4] = detections[inds, :4]
                            cls_det[:, 4] = detections[inds, 4] * detections[inds, 5]
                            all_boxes[cls][img_id] = cls_det.cpu().numpy()

                if vis:
                    # print(val_imdb.image_path_at(img_id))
                    img = Image.open(val_imdb.image_path_at(img_id)).convert("RGB")
                    if len(detections) == 0:
                        continue
                    det_boxes = detections[:, :5].cpu().numpy()
                    det_classes = detections[:, -1].long().cpu().numpy()
                    out_path = "./Visualize"
                    Path(out_path).mkdir(parents=True, exist_ok=True)
                    # img.save(f'{out_path}/E{epoch}_B{batch}_I{i}_Original.jpg')
                    im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=val_imdb.classes)
                    im2show.save(f'{out_path}/E{epoch}_B{batch}_I{i}_Output.jpg')

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    val_imdb.evaluate_detections(all_boxes, output_dir=output_dir)
    
    
def detection(yolo_outputs, epoch, val_imdb, small_val_dataloader):
    conf_thresh = 0.5
    nms_thresh = 0.5
    batch_size = 2
    output_dir = "/home/msis/Desktop/pcie_python/GUI_List_Iter/Output_data"
    dataset = "voc07test"
    vis = True

    dataset_size = len(val_imdb.image_index)

    all_boxes = [[[] for _ in range(dataset_size)] for _ in range(val_imdb.num_classes)]

    det_file = os.path.join(output_dir, 'detections.pkl')

    img_id = -1

    with torch.no_grad():
        for batch, (im_data, im_infos) in enumerate(small_val_dataloader):
            for i in range(im_data.size(0)):
                img_id += 1
                output = [item[i].data for item in yolo_outputs]
                im_info = {'width': im_infos[i][0], 'height': im_infos[i][1]}
                detections = yolo_eval(output, im_info, conf_threshold=conf_thresh, nms_threshold=nms_thresh)

                if len(detections) > 0:
                    for cls in range(val_imdb.num_classes):
                        inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                        if inds.numel() > 0:
                            cls_det = torch.zeros((inds.numel(), 5))
                            cls_det[:, :4] = detections[inds, :4]
                            cls_det[:, 4] = detections[inds, 4] * detections[inds, 5]
                            all_boxes[cls][img_id] = cls_det.cpu().numpy()

                if vis:
                    img = Image.open(val_imdb.image_path_at(img_id)).convert("RGB")
                    out_path = "/home/msis/Desktop/pcie_python/GUI_List_Iter/Output_data/Visualize"
                    Path(out_path).mkdir(parents=True, exist_ok=True)
                    if len(detections) == 0:
                        continue
                    det_boxes = detections[:, :5].cpu().numpy()
                    det_classes = detections[:, -1].long().cpu().numpy()
                    img.save(f'{out_path}/E{epoch}_B{batch}_I{i}_Original.jpg')
                    im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=val_imdb.classes)
                    im2show.save(f'{out_path}/E{epoch}_B{batch}_I{i}_Output.jpg')

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    val_imdb.evaluate_detections(all_boxes, output_dir=output_dir)

if __name__ == '__main__':
    Detection(160)















