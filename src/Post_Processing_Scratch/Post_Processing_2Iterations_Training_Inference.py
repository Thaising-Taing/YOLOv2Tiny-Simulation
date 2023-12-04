from Post_Processing_Scratch.Post_Processing_Function import *
from Post_Processing_Scratch.Calculate_Loss_2Iterations import *
import os
import argparse
import time
import numpy as np
import pickle
import torch
from torch.autograd import Variable
from PIL import Image
from Post_Processing_Scratch.dataset.factory import get_imdb
from Post_Processing_Scratch.dataset.roidb import RoiDataset
from Post_Processing_Scratch.util.visualize import draw_detection_boxes
from Post_Processing_Scratch.dataset.roidb import RoiDataset, detection_collate
import matplotlib.pyplot as plt
from Post_Processing_Scratch.util.network import WeightLoader
from torch.utils.data import DataLoader
from Post_Processing_Scratch.config import config as cfg
from tqdm import tqdm


# Conditions for Reading Original_Data from Hardware Processing:
class Post_Processing_Inference:
    def __init__(self, Mode, Brain_Floating_Point, Exponent_Bits, Mantissa_Bits, OutImage1_Data_CH0, OutImage1_Data_CH1,
                 OutImage2_Data_CH0, OutImage2_Data_CH1, OutImage3_Data_CH0, OutImage3_Data_CH1, 
                 OutImage4_Data_CH0, OutImage4_Data_CH1, OutImage5_Data_CH0, OutImage5_Data_CH1,
                 OutImage6_Data_CH0, OutImage6_Data_CH1, OutImage7_Data_CH0, OutImage7_Data_CH1,
                 OutImage8_Data_CH0, OutImage8_Data_CH1):
        self.Mode = Mode
        self.Brain_Floating_Point = Brain_Floating_Point
        self.Exponent_Bits = Exponent_Bits
        self.Mantissa_Bits = Mantissa_Bits
        self.OutImage1_Data_CH0 = OutImage1_Data_CH0
        self.OutImage1_Data_CH1 = OutImage1_Data_CH1
        self.OutImage2_Data_CH0 = OutImage2_Data_CH0
        self.OutImage2_Data_CH1 = OutImage2_Data_CH1
        self.OutImage3_Data_CH0 = OutImage3_Data_CH0
        self.OutImage3_Data_CH1 = OutImage3_Data_CH1
        self.OutImage4_Data_CH0 = OutImage4_Data_CH0
        self.OutImage4_Data_CH1 = OutImage4_Data_CH1
        self.OutImage5_Data_CH0 = OutImage5_Data_CH0
        self.OutImage5_Data_CH1 = OutImage5_Data_CH1
        self.OutImage6_Data_CH0 = OutImage6_Data_CH0
        self.OutImage6_Data_CH1 = OutImage6_Data_CH1
        self.OutImage7_Data_CH0 = OutImage7_Data_CH0
        self.OutImage7_Data_CH1 = OutImage7_Data_CH1
        self.OutImage8_Data_CH0 = OutImage8_Data_CH0
        self.OutImage8_Data_CH1 = OutImage8_Data_CH1
        
    def PostProcessing_Inference(self, gt_boxes, gt_classes, num_boxes):
        global Layer8_Loss_Gradient, Numerical_Loss
        
        if self.Brain_Floating_Point:
            Output_Image1 = OutFmap_Layer8_BFPtoDec(self.OutImage1_Data_CH0, self.OutImage1_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image2 = OutFmap_Layer8_BFPtoDec(self.OutImage2_Data_CH0, self.OutImage2_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image3 = OutFmap_Layer8_BFPtoDec(self.OutImage3_Data_CH0, self.OutImage3_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image4 = OutFmap_Layer8_BFPtoDec(self.OutImage4_Data_CH0, self.OutImage4_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image5 = OutFmap_Layer8_BFPtoDec(self.OutImage5_Data_CH0, self.OutImage5_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image6 = OutFmap_Layer8_BFPtoDec(self.OutImage6_Data_CH0, self.OutImage6_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image7 = OutFmap_Layer8_BFPtoDec(self.OutImage7_Data_CH0, self.OutImage7_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image8 = OutFmap_Layer8_BFPtoDec(self.OutImage8_Data_CH0, self.OutImage8_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image = Output_Image1 + Output_Image2 + Output_Image3 + Output_Image4 + \
                           Output_Image5 + Output_Image6 + Output_Image7 + Output_Image8
        else:
            Output_Image1 = OutFmap_Layer8_FP32toDec(self.OutImage1_Data_CH0, self.OutImage1_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image2 = OutFmap_Layer8_FP32toDec(self.OutImage2_Data_CH0, self.OutImage2_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image3 = OutFmap_Layer8_FP32toDec(self.OutImage3_Data_CH0, self.OutImage3_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image4 = OutFmap_Layer8_FP32toDec(self.OutImage4_Data_CH0, self.OutImage4_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image5 = OutFmap_Layer8_FP32toDec(self.OutImage5_Data_CH0, self.OutImage5_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image6 = OutFmap_Layer8_FP32toDec(self.OutImage6_Data_CH0, self.OutImage6_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image7 = OutFmap_Layer8_FP32toDec(self.OutImage7_Data_CH0, self.OutImage7_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image8 = OutFmap_Layer8_FP32toDec(self.OutImage8_Data_CH0, self.OutImage8_Data_CH1, self.Exponent_Bits, self.Mantissa_Bits)
            Output_Image = Output_Image1 + Output_Image2 + Output_Image3 + Output_Image4 + \
                           Output_Image5 + Output_Image6 + Output_Image7 + Output_Image8

        _data = Output_Image
        output_file = os.path.join("Output_data", f'output.pickle')
        with open(output_file, 'wb') as handle:
            pickle.dump(_data, handle, protocol=pickle.HIGHEST_PROTOCOL)  

        print(Output_Image)       

        # Mode is Training
        if self.Mode == "Training":
            # Convert Output Image into List of Floating 32 Format
            Float_OutputImage = [np.float32(x) for x in Output_Image]
            # Loss Calculation and Loss Gradient Calculation
            Layer8_Loss, Layer8_Loss_Gradient = loss(Float_OutputImage, gt_boxes=gt_boxes,
                                                     gt_classes=gt_classes, num_boxes=num_boxes)
            # Numerical_Loss = Numerical_Loss(Layer8_Loss)
            return Layer8_Loss, Layer8_Loss_Gradient
            
        # Mode is Inference   
        if self.Mode == "Inference":
            Float_OutputImage = [np.float32(x) for x in Output_Image]
            output_data = reshape_output(Float_OutputImage)
            return output_data
            # Float_OutputImage = [np.float32(x) for x in Output_Image]
            # target_data, output_data = reshape_output(Float_OutputImage, gt_boxes=gt_boxes,
            #                                          gt_classes=gt_classes, num_boxes=num_boxes)
            # return target_data, output_data

def reshape_output(out):
    # print('Calculating the loss and its gradients for python model.')
    out = torch.tensor(out, requires_grad=True)
    
    # Additional Condition: 
    out = out[0:(8*125*(13**2))]

    out = out.reshape(8, 125, 13, 13)
    scores = out
    bsize, _, h, w = out.shape
    out = out.permute(0, 2, 3, 1).contiguous().view(bsize, 13 * 13 * 5, 5 + 20)

    xy_pred = torch.sigmoid(out[:, :, 0:2])
    conf_pred = torch.sigmoid(out[:, :, 4:5])
    hw_pred = torch.exp(out[:, :, 2:4])
    class_score = out[:, :, 5:]
    class_pred = F.softmax(class_score, dim=-1)
    delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

    output_variable = (delta_pred, conf_pred, class_score)
    output_data = [v.data for v in output_variable]

    # return output_data
    return output_variable



def scale_boxes(boxes, im_info):
    """
    scale predicted boxes

    Arguments:
    boxes -- tensor of shape (N, 4) xxyy format
    im_info -- dictionary {width:, height:}

    Returns:
    scaled_boxes -- tensor of shape (N, 4) xxyy format

    """

    h = im_info['height']
    w = im_info['width']

    input_h, input_w = cfg.test_input_size
    scale_h, scale_w = input_h / h, input_w / w

    # scale the boxes
    boxes *= cfg.strides

    boxes[:, 0::2] /= scale_w
    boxes[:, 1::2] /= scale_h

    boxes = xywh2xxyy(boxes)

    # clamp boxes
    boxes[:, 0::2].clamp_(0, w-1)
    boxes[:, 1::2].clamp_(0, h-1)

    return boxes

def yolo_filter_boxes(boxes_pred, conf_pred, classes_pred, confidence_threshold=0.6):
    """
    Filter boxes whose confidence is lower than a given threshold

    Arguments:
    boxes_pred -- tensor of shape (H * W * num_anchors, 4) (x1, y1, x2, y2) predicted boxes
    conf_pred -- tensor of shape (H * W * num_anchors, 1)
    classes_pred -- tensor of shape (H * W * num_anchors, num_classes)
    threshold -- float, threshold used to filter boxes

    Returns:
    filtered_boxes -- tensor of shape (num_positive, 4)
    filtered_conf -- tensor of shape (num_positive, 1)
    filtered_cls_max_conf -- tensor of shape (num_positive, num_classes)
    filtered_cls_max_id -- tensor of shape (num_positive, num_classes)
    """

    # multiply class scores and objectiveness score
    # use class confidence score
    # TODO: use objectiveness (IOU) score or class confidence score
    cls_max_conf, cls_max_id = torch.max(classes_pred, dim=-1, keepdim=True)
    cls_conf = conf_pred * cls_max_conf

    pos_inds = (cls_conf > confidence_threshold).view(-1)

    filtered_boxes = boxes_pred[pos_inds, :]

    filtered_conf = conf_pred[pos_inds, :]

    filtered_cls_max_conf = cls_max_conf[pos_inds, :]

    filtered_cls_max_id = cls_max_id[pos_inds, :]

    return filtered_boxes, filtered_conf, filtered_cls_max_conf, filtered_cls_max_id.float()

def generate_prediction_boxes(deltas_pred):
    """
    Apply deltas prediction to pre-defined anchors

    Arguments:
    deltas_pred -- tensor of shape (H * W * num_anchors, 4) σ(t_x), σ(t_y), σ(t_w), σ(t_h)

    Returns:
    boxes_pred -- tensor of shape (H * W * num_anchors, 4)  (x1, y1, x2, y2)
    """
    H = int(cfg.test_input_size[0] / cfg.strides)
    W = int(cfg.test_input_size[1] / cfg.strides)
    anchors = torch.FloatTensor(cfg.anchors)
    all_anchors_xywh = generate_all_anchors(anchors, H, W) # shape: (H * W * num_anchors, 4), format: (x, y, w, h)
    all_anchors_xywh = deltas_pred.new(*all_anchors_xywh.size()).copy_(all_anchors_xywh)
    boxes_pred = box_transform_inv(all_anchors_xywh, deltas_pred)

    return boxes_pred

def yolo_nms(boxes, scores, threshold):
    """
    Apply Non-Maximum-Suppression on boxes according to their scores

    Arguments:
    boxes -- tensor of shape (N, 4) (x1, y1, x2, y2)
    scores -- tensor of shape (N) confidence
    threshold -- float. NMS threshold

    Returns:
    keep -- tensor of shape (None), index of boxes which should be retain.
    """

    score_sort_index = torch.sort(scores, dim=0, descending=True)[1]

    keep = []

    while score_sort_index.numel() > 0:

        i = score_sort_index[0]
        keep.append(i)

        if score_sort_index.numel() == 1:
            break

        cur_box = boxes[score_sort_index[0], :].view(-1, 4)
        res_box = boxes[score_sort_index[1:], :].view(-1, 4)

        ious = box_ious(cur_box, res_box).view(-1)

        inds = torch.nonzero(ious < threshold).squeeze()

        score_sort_index = score_sort_index[inds + 1].view(-1)

    return torch.LongTensor(keep)

def yolo_eval(yolo_output, im_info, conf_threshold=0.6, nms_threshold=0.4):
    """
    Evaluate the yolo output, generate the final predicted boxes

    Arguments:
    yolo_output -- list of tensors (deltas_pred, conf_pred, classes_pred)

    deltas_pred -- tensor of shape (H * W * num_anchors, 4) σ(t_x), σ(t_y), σ(t_w), σ(t_h)
    conf_pred -- tensor of shape (H * W * num_anchors, 1)
    classes_pred -- tensor of shape (H * W * num_anchors, num_classes)

    im_info -- dictionary {w:, h:}

    threshold -- float, threshold used to filter boxes


    Returns:
    detections -- tensor of shape (None, 7) (x1, y1, x2, y2, cls_conf, cls)
    """

    deltas = yolo_output[0].cpu()
    conf = yolo_output[1].cpu()
    classes = yolo_output[2].cpu()

    num_classes = classes.size(1)
    # apply deltas to anchors

    boxes = generate_prediction_boxes(deltas)

    if cfg.debug:
        print('check box: ', boxes.view(13*13, 5, 4).permute(1, 0, 2).contiguous().view(-1,4)[0:10,:])
        print('check conf: ', conf.view(13*13, 5).permute(1,0).contiguous().view(-1)[:10])

    # filter boxes on confidence score
    boxes, conf, cls_max_conf, cls_max_id = yolo_filter_boxes(boxes, conf, classes, conf_threshold)

    # no detection !
    if boxes.size(0) == 0:
        return []

    # scale boxes
    boxes = scale_boxes(boxes, im_info)

    if cfg.debug:
        all_boxes = torch.cat([boxes, conf, cls_max_conf, cls_max_id], dim=1)
        print('check all boxes: ', all_boxes)
        print('check all boxes len: ', len(all_boxes))
    #
    # apply nms
    # keep = yolo_nms(boxes, conf.view(-1), nms_threshold)
    # boxes_keep = boxes[keep, :]
    # conf_keep = conf[keep, :]
    # cls_max_conf = cls_max_conf[keep, :]
    # cls_max_id = cls_max_id.view(-1, 1)[keep, :]
    #
    # if cfg.debug:
    #     print('check nms all boxes len: ', len(boxes_keep))
    #
    # seq = [boxes_keep, conf_keep, cls_max_conf, cls_max_id.float()]
    #
    # return torch.cat(seq, dim=1)

    detections = []

    cls_max_id = cls_max_id.view(-1)

    # apply NMS classwise
    for cls in range(num_classes):
        cls_mask = cls_max_id == cls
        inds = torch.nonzero(cls_mask).squeeze()

        if inds.numel() == 0:
            continue

        boxes_pred_class = boxes[inds, :].view(-1, 4)
        conf_pred_class = conf[inds, :].view(-1, 1)
        cls_max_conf_class = cls_max_conf[inds].view(-1, 1)
        classes_class = cls_max_id[inds].view(-1, 1)

        nms_keep = yolo_nms(boxes_pred_class, conf_pred_class.view(-1), nms_threshold)

        boxes_pred_class_keep = boxes_pred_class[nms_keep, :]
        conf_pred_class_keep = conf_pred_class[nms_keep, :]
        cls_max_conf_class_keep = cls_max_conf_class.view(-1, 1)[nms_keep, :]
        classes_class_keep = classes_class.view(-1, 1)[nms_keep, :]

        seq = [boxes_pred_class_keep, conf_pred_class_keep, cls_max_conf_class_keep, classes_class_keep.float()]

        detections_cls = torch.cat(seq, dim=-1)
        detections.append(detections_cls)

    return torch.cat(detections, dim=0)
 
def Detection(output_data, epoch, small_val_dataloader, val_imdb):
# def Detection(output_data, small_val_dataloader, val_imdb):
    print(output_data)
    _data = output_data, epoch
    output_file = os.path.join("Outdata_error", f'Params_{epoch}.pickle')
    with open(output_file, 'wb') as handle:
        pickle.dump(_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    dataset_size = len(val_imdb.image_index)

    all_boxes = [[[] for _ in range(dataset_size)] for _ in range(val_imdb.num_classes)]

    img_id = -1
    with torch.no_grad():
        for batch, (im_data, im_infos) in enumerate(small_val_dataloader):
            for i in range(im_data.size(0)):
                img_id += 1
                im_info = {'width': im_infos[i][0], 'height': im_infos[i][1]}
                detections = yolo_eval(output_data, im_info, conf_threshold=0.5, nms_threshold=0.5)
                # print('im detect [{}/{}]'.format(img_id+1, len(val_dataset)))
                if len(detections) > 0:
                    for cls in range(val_imdb.num_classes):
                        inds = torch.nonzero(detections[:, -1] == cls).view(-1)
                        if inds.numel() > 0:
                            cls_det = torch.zeros((inds.numel(), 5))
                            cls_det[:, :4] = detections[inds, :4]
                            cls_det[:, 4] = detections[inds, 4] * detections[inds, 5]
                            all_boxes[cls][img_id] = cls_det.cpu().numpy()

                
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
                # im2show.save(f'{out_path}/B{batch}_I{i}_Output.jpg')


# def Detection(target_data, output_data, epoch, small_val_dataloader, val_imdb):
    
#     dataset_size = len(val_imdb.image_index)

#     all_boxes = [[[] for _ in range(dataset_size)] for _ in range(val_imdb.num_classes)]

#     img_id = -1
#     with torch.no_grad():
#         for batch, (im_data, im_infos) in enumerate(small_val_dataloader):

#             for i in range(im_data.size(0)):
#                 img_id += 1
#                 im_info = {'width': im_infos[i][0], 'height': im_infos[i][1]}
#                 detections = yolo_eval(output_data, im_info, conf_threshold=0.5, nms_threshold=0.5)
#                 # print('im detect [{}/{}]'.format(img_id+1, len(val_dataset)))
#                 if len(detections) > 0:
#                     for cls in range(val_imdb.num_classes):
#                         inds = torch.nonzero(detections[:, -1] == cls).view(-1)
#                         if inds.numel() > 0:
#                             cls_det = torch.zeros((inds.numel(), 5))
#                             cls_det[:, :4] = detections[inds, :4]
#                             cls_det[:, 4] = detections[inds, 4] * detections[inds, 5]
#                             all_boxes[cls][img_id] = cls_det.cpu().numpy()


#                 img = Image.open(val_imdb.image_path_at(img_id)).convert("RGB")
#                 if len(detections) == 0:
#                     continue
#                 det_boxes = detections[:, :5].cpu().numpy()
#                 det_classes = detections[:, -1].long().cpu().numpy()
#                 out_path = "./Visualize"
#                 Path(out_path).mkdir(parents=True, exist_ok=True)
#                 # img.save(f'{out_path}/E{epoch}_B{batch}_I{i}_Original.jpg')
#                 im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=val_imdb.classes)
#                 im2show.save(f'{out_path}/E{epoch}_B{batch}_I{i}_Output.jpg')
                    
                # plt.figure()
                # plt.imshow(im2show)
                # plt.show()

         
    # Define the image shape:
    # h,w,c = im_data.shape
    # dataset_size = len(val_imdb.image_index)
    # all_boxes = [[[] for _ in range(dataset_size)] for _ in range(val_imdb.num_classes)]
    # img_id = -1
    # im_info = {'width': w, 'height': h}
    # detections = yolo_eval(output_data, im_info, conf_threshold=0.5,nms_threshold=0.5)

    # if len(detections) > 0:
    #     for cls in range(val_imdb.num_classes):
    #         inds = torch.nonzero(detections[:, -1] == cls).view(-1)
    #         if inds.numel() > 0:
    #             cls_det = torch.zeros((inds.numel(), 5))
    #             cls_det[:, :4] = detections[inds, :4]
    #             cls_det[:, 4] = detections[inds, 4] * detections[inds, 5]
    #             all_boxes[cls][img_id] = cls_det.cpu().numpy()
                
    # # Load an image using PIL
    # img = Image.open(val_imdb.image_path_at(img_id)).convert("RGB")
    
    # # if len(detections) == 0:
    # #     continue
    # det_boxes = detections[:, :5].cpu().numpy()
    # det_classes = detections[:, -1].long().cpu().numpy()
    # out_path = "./Visualize"
    # Path(out_path).mkdir(parents=True, exist_ok=True)
    # # img.save(f'{out_path}/E{epoch}_B{batch}_I{i}_Original.jpg')
    # im2show = draw_detection_boxes(img, det_boxes, det_classes, class_names=val_imdb.classes)
    # im2show.save(f'{out_path}/E{epoch}_B{batch}_I{i}_Output.jpg')
                

