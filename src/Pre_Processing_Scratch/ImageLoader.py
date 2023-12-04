# from Pre_Processing_Scratch.cnn_python_latest_LightNorm import *
import sys 
sys.path.append("../")
from torch.utils.data import DataLoader
from Dataset.factory import get_imdb
from Dataset.roidb import RoiDataset, detection_collate
from config import config as cfg
import torch
import numpy as np
import pickle
from pathlib import Path

def prepare_im_data(img):

    im_info = dict()
    im_info['width'], im_info['height'] = img.size

    # resize the image
    H, W = cfg.input_size
    im_data = img.resize((H, W))

    # to torch tensor
    im_data = torch.from_numpy(np.array(im_data)).float() / 255

    im_data = im_data.permute(2, 0, 1).unsqueeze(0)

    return im_data, im_info

def ImageLoader():
    _Dataset = False
    if _Dataset:
        dataset = 'voc0712trainval'
        imdb_name = 'voc_2007_trainval+voc_2012_trainval'
        imdbval_name = 'voc_2007_test'

        def axpy(N: int = 0., ALPHA: float = 1., X: int = 0, INCX: int = 1, Y: float = 0., INCY: int = 1):
            for i in range(N):
                Y[i * INCY] += ALPHA * X[i * INCX]

        def get_dataset(datasetnames):
            names = datasetnames.split('+')
            dataset = RoiDataset(get_imdb(names[0]))
            print('load dataset {}'.format(names[0]))
            for name in names[1:]:
                tmp = RoiDataset(get_imdb(name))
                dataset += tmp
                # print('load and add dataset {}'.format(name))
            return dataset

        train_dataset = get_dataset(imdb_name)

    _Dataloader = _Dataset
    if _Dataloader:
        train_dataloader = DataLoader(train_dataset, batch_size=8,
                                      shuffle=True, num_workers=2,
                                      collate_fn=detection_collate, drop_last=True)
        train_data_iter = iter(train_dataloader)

    _Get_Next_Data = _Dataloader
    if _Get_Next_Data:
        _data = next(train_data_iter)
        im_data, gt_boxes, gt_classes, num_obj = _data

        __data = im_data, gt_boxes, gt_classes, num_obj

        Path("Temp_Files").mkdir(parents=True, exist_ok=True)

        with open('/home/msis/Desktop/pcie_python/GUI_list/Pre_Processing_Scratch/data/pretrained/Input_Data_Batch8.pickle', 'wb') as handle:
            pickle.dump(_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('/home/msis/Desktop/pcie_python/GUI_list/Pre_Processing_Scratch/data/pretrained/Input_Data_Batch8.pickle', 'rb') as handle:
            b = pickle.load(handle)
        im_data, gt_boxes, gt_classes, num_obj = b
        __data = im_data, gt_boxes, gt_classes, num_obj

    return im_data
