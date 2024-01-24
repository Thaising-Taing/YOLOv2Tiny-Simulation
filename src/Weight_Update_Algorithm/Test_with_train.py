from tqdm import tqdm
import os
import argparse
import time
import numpy as np
import pickle
import torch
from torch.autograd import Variable
from pathlib import Path
from PIL import Image
import os
from Weight_Update_Algorithm.yolov2tiny_LightNorm_2Iterations import Yolov2
from Dataset.factory import get_imdb
from Dataset.roidb import RoiDataset
from Weight_Update_Algorithm.yolo_eval import yolo_eval
from src.util.visualize import draw_detection_boxes
import matplotlib.pyplot as plt
from src.util.network import WeightLoader
from torch.utils.data import DataLoader
from my_config import config as cfg


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


def test_for_train(temp_path, model, args, val_dataloader=[], val_dataset=[], val_imdb=[]):
    # args.dataset = "voc07test"
    args.conf_thresh = 0.001
    if args.vis:
        args.conf_thresh = 0.5
    args.nms_thresh = 0.45
    
    args.output_dir = temp_path
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if val_dataloader==[]:
        # prepare dataset
        args.imdbval_name = 'voc_2007_test'
        # args.imdbval_name = 'voc_2007_test-car'
        val_imdb = get_imdb(args.imdbval_name)

        val_dataset = RoiDataset(val_imdb, train=False)
        # if args.use_small_dataset: args.data_limit = 80
        # if not args.data_limit==0:
        #     val_dataset = torch.utils.data.Subset(val_dataset, range(0, args.data_limit))
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size*8, shuffle=False, drop_last=True, num_workers=args.num_workers, persistent_workers=True)

    # load model
    # model = Yolov2()
    # model = model()

    # model_path = os.path.join(args.output_dir, 'weights.pth')
    # torch.save({'model': model.state_dict(),} , model_path)
    # if torch.cuda.is_available():
    #     checkpoint = torch.load(weights)
    # else:
    #     checkpoint = torch.load(weights, map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    # # print(f'Model loaded from {model_path}')

    if args.use_cuda:
        model.cuda()

    model.eval()

    dataset_size = len(val_imdb.image_index)
    # dataset_size = len(val_dataset)

    all_boxes = [[[] for _ in range(dataset_size)] for _ in range(val_imdb.num_classes)]

    args.output_dir = os.path.join(args.output_dir, "Outputs")
    os.makedirs( args.output_dir, exist_ok=True )
    det_file = os.path.join(args.output_dir, 'detections.pkl')

    img_id = -1
    with torch.no_grad():
        for batch, (im_data, im_infos) in tqdm(enumerate(val_dataloader), total=int(dataset_size/args.batch_size), desc="Performing validation with {} images".format(dataset_size), leave=False):
        # for batch, (im_data, im_infos) in enumerate(val_dataloader):
        # for batch, (im_data, im_infos) in enumerate(small_val_dataloader):
            if args.use_cuda:
                im_data_variable = Variable(im_data).cuda()
            else:
                im_data_variable = Variable(im_data)

            yolo_outputs = model(im_data_variable)
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


    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    # map = val_imdb.evaluate_detections(all_boxes, output_dir=args.output_dir)
    map = val_imdb.evaluate_detections_with_train(all_boxes, output_dir=args.output_dir)
    return map
















