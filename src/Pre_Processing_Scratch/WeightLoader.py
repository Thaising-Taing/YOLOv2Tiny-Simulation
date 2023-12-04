import numpy as np
from Pre_Processing_Scratch.DeepConvNet import *
import torch.nn as nn
import torch.nn.functional as F
from config import config as cfg
# from Pre_Processing_Scratch.cnn_python_latest_LightNorm import build_target, yolo_loss

def box_ious(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2 (x1, y1, x2, y2)

    Arguments:
    box1 -- tensor of shape (N, 4), first set of boxes
    box2 -- tensor of shape (K, 4), second set of boxes

    Returns:
    ious -- tensor of shape (N, K), ious between boxes
    """

    N = box1.size(0)
    K = box2.size(0)

    # when torch.max() takes tensor of different shape as arguments, it will broadcasting them.
    xi1 = torch.max(box1[:, 0].view(N, 1), box2[:, 0].view(1, K))
    yi1 = torch.max(box1[:, 1].view(N, 1), box2[:, 1].view(1, K))
    xi2 = torch.min(box1[:, 2].view(N, 1), box2[:, 2].view(1, K))
    yi2 = torch.min(box1[:, 3].view(N, 1), box2[:, 3].view(1, K))

    # we want to compare the compare the value with 0 elementwise. However, we can't
    # simply feed int 0, because it will invoke the function torch(max, dim=int) which is not
    # what we want.
    # To feed a tensor 0 of same type and device with box1 and box2
    # we use tensor.new().fill_(0)

    iw = torch.max(xi2 - xi1, box1.new(1).fill_(0))
    ih = torch.max(yi2 - yi1, box1.new(1).fill_(0))

    inter = iw * ih

    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    box1_area = box1_area.view(N, 1)
    box2_area = box2_area.view(1, K)

    union_area = box1_area + box2_area - inter

    ious = inter / union_area

    return ious


def xxyy2xywh(box):
    """
    Convert the box (x1, y1, x2, y2) encoding format to (c_x, c_y, w, h) format

    Arguments:
    box: tensor of shape (N, 4), boxes of (x1, y1, x2, y2) format

    Returns:
    xywh_box: tensor of shape (N, 4), boxes of (c_x, c_y, w, h) format
    """

    c_x = (box[:, 2] + box[:, 0]) / 2
    c_y = (box[:, 3] + box[:, 1]) / 2
    w = box[:, 2] - box[:, 0]
    h = box[:, 3] - box[:, 1]

    c_x = c_x.view(-1, 1)
    c_y = c_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    xywh_box = torch.cat([c_x, c_y, w, h], dim=1)
    return xywh_box


def xywh2xxyy(box):
    """
    Convert the box encoding format form (c_x, c_y, w, h) to (x1, y1, x2, y2)

    Arguments:
    box -- tensor of shape (N, 4), box of (c_x, c_y, w, h) format

    Returns:
    xxyy_box -- tensor of shape (N, 4), box of (x1, y1, x2, y2) format
    """

    x1 = box[:, 0] - (box[:, 2]) / 2
    y1 = box[:, 1] - (box[:, 3]) / 2
    x2 = box[:, 0] + (box[:, 2]) / 2
    y2 = box[:, 1] + (box[:, 3]) / 2

    x1 = x1.view(-1, 1)
    y1 = y1.view(-1, 1)
    x2 = x2.view(-1, 1)
    y2 = y2.view(-1, 1)

    xxyy_box = torch.cat([x1, y1, x2, y2], dim=1)
    return xxyy_box


def box_transform(box1, box2):
    """
    Calculate the delta values σ(t_x), σ(t_y), exp(t_w), exp(t_h) used for transforming box1 to  box2

    Arguments:
    box1 -- tensor of shape (N, 4) first set of boxes (c_x, c_y, w, h)
    box2 -- tensor of shape (N, 4) second set of boxes (c_x, c_y, w, h)

    Returns:
    deltas -- tensor of shape (N, 4) delta values (t_x, t_y, t_w, t_h)
                   used for transforming boxes to reference boxes
    """

    t_x = box2[:, 0] - box1[:, 0]
    t_y = box2[:, 1] - box1[:, 1]
    t_w = box2[:, 2] / box1[:, 2]
    t_h = box2[:, 3] / box1[:, 3]

    t_x = t_x.view(-1, 1)
    t_y = t_y.view(-1, 1)
    t_w = t_w.view(-1, 1)
    t_h = t_h.view(-1, 1)

    # σ(t_x), σ(t_y), exp(t_w), exp(t_h)
    deltas = torch.cat([t_x, t_y, t_w, t_h], dim=1)
    return deltas


def box_transform_inv(box, deltas):
    """
    apply deltas to box to generate predicted boxes

    Arguments:
    box -- tensor of shape (N, 4), boxes, (c_x, c_y, w, h)
    deltas -- tensor of shape (N, 4), deltas, (σ(t_x), σ(t_y), exp(t_w), exp(t_h))

    Returns:
    pred_box -- tensor of shape (N, 4), predicted boxes, (c_x, c_y, w, h)
    """

    c_x = box[:, 0] + deltas[:, 0]
    c_y = box[:, 1] + deltas[:, 1]
    w = box[:, 2] * deltas[:, 2]
    h = box[:, 3] * deltas[:, 3]

    c_x = c_x.view(-1, 1)
    c_y = c_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    pred_box = torch.cat([c_x, c_y, w, h], dim=-1)
    return pred_box

def generate_all_anchors(anchors, H, W):

    # number of anchors per cell
    A = anchors.size(0)

    # number of cells
    K = H * W

    shift_x, shift_y = torch.meshgrid([torch.arange(0, W), torch.arange(0, H)])

    # transpose shift_x and shift_y because we want our anchors to be organized in H x W order
    shift_x = shift_x.t().contiguous()
    shift_y = shift_y.t().contiguous()

    # shift_x is a long tensor, c_x is a float tensor
    c_x = shift_x.float()
    c_y = shift_y.float()

    centers = torch.cat([c_x.view(-1, 1), c_y.view(-1, 1)], dim=-1)  # tensor of shape (h * w, 2), (cx, cy)

    # add anchors width and height to centers
    all_anchors = torch.cat([centers.view(K, 1, 2).expand(K, A, 2),
                             anchors.view(1, A, 2).expand(K, A, 2)], dim=-1)

    all_anchors = all_anchors.view(-1, 4)

    return all_anchors

def yolo_loss(output, target):

    delta_pred_batch = output[0]
    conf_pred_batch = output[1]
    class_score_batch = output[2]

    iou_target = target[0]
    iou_mask = target[1]
    box_target = target[2]
    box_mask = target[3]
    class_target = target[4]
    class_mask = target[5]

    b, _, num_classes = class_score_batch.size()
    class_score_batch = class_score_batch.view(-1, num_classes)
    class_target = class_target.view(-1)
    class_mask = class_mask.view(-1)

    # ignore the gradient of noobject's target
    class_keep = class_mask.nonzero().squeeze(1)
    class_score_batch_keep = class_score_batch[class_keep, :]
    class_target_keep = class_target[class_keep]

    # if cfg.debug:
    #     print(class_score_batch_keep)
    #     print(class_target_keep)

    # calculate the loss, normalized by batch size.
    box_loss = 1 / b * cfg.coord_scale * F.mse_loss(delta_pred_batch * box_mask, box_target * box_mask,
                                                    reduction='sum') / 2.0
    iou_loss = 1 / b * F.mse_loss(conf_pred_batch * iou_mask, iou_target * iou_mask, reduction='sum') / 2.0
    class_loss = 1 / b * cfg.class_scale * F.cross_entropy(class_score_batch_keep, class_target_keep, reduction='sum')

    return box_loss, iou_loss, class_loss

def build_target(output, gt_data, H, W):
    
    delta_pred_batch = output[0]
    conf_pred_batch = output[1]
    class_score_batch = output[2]

    gt_boxes_batch = gt_data[0]
    gt_classes_batch = gt_data[1]
    num_boxes_batch = gt_data[2]

    bsize = delta_pred_batch.size(0)

    num_anchors = 5  # hard code for now

    # initial the output tensor
    # we use `tensor.new()` to make the created tensor has the same devices and data type as input tensor's
    # what tensor is used doesn't matter
    iou_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    iou_mask = delta_pred_batch.new_ones((bsize, H * W, num_anchors, 1)) * cfg.noobject_scale

    box_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 4))
    box_mask = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

    class_target = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    class_mask = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

    # get all the anchors

    anchors = torch.FloatTensor(cfg.anchors)

    # note: the all anchors' xywh scale is normalized by the grid width and height, i.e. 13 x 13
    # this is very crucial because the predict output is normalized to 0~1, which is also
    # normalized by the grid width and height
    all_grid_xywh = generate_all_anchors(anchors, H, W)  # shape: (H * W * num_anchors, 4), format: (x, y, w, h)
    all_grid_xywh = delta_pred_batch.new(*all_grid_xywh.size()).copy_(all_grid_xywh)
    all_anchors_xywh = all_grid_xywh.clone()
    all_anchors_xywh[:, 0:2] += 0.5
    if cfg.debug:
        print('all grid: ', all_grid_xywh[:12, :])
        print('all anchor: ', all_anchors_xywh[:12, :])
    all_anchors_xxyy = xywh2xxyy(all_anchors_xywh)

    # process over batches
    for b in range(bsize):
        num_obj = num_boxes_batch[b].item()
        delta_pred = delta_pred_batch[b]
        gt_boxes = gt_boxes_batch[b][:num_obj, :]
        gt_classes = gt_classes_batch[b][:num_obj]

        # rescale ground truth boxes
        gt_boxes[:, 0::2] *= W
        gt_boxes[:, 1::2] *= H

        # step 1: process IoU target

        # apply delta_pred to pre-defined anchors
        all_anchors_xywh = all_anchors_xywh.view(-1, 4)
        box_pred = box_transform_inv(all_grid_xywh, delta_pred)
        box_pred = xywh2xxyy(box_pred)

        # for each anchor, its iou target is corresponded to the max iou with any gt boxes
        ious = box_ious(box_pred, gt_boxes)  # shape: (H * W * num_anchors, num_obj)
        ious = ious.view(-1, num_anchors, num_obj)
        max_iou, _ = torch.max(ious, dim=-1, keepdim=True)  # shape: (H * W, num_anchors, 1)
        if cfg.debug:
            print('ious', ious)

        # iou_target[b] = max_iou

        # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold
        iou_thresh_filter = max_iou.view(-1) > cfg.thresh
        n_pos = torch.nonzero(iou_thresh_filter).numel()

        if n_pos > 0:
            iou_mask[b][max_iou >= cfg.thresh] = 0

        # step 2: process box target and class target
        # calculate overlaps between anchors and gt boxes
        overlaps = box_ious(all_anchors_xxyy, gt_boxes).view(-1, num_anchors, num_obj)
        gt_boxes_xywh = xxyy2xywh(gt_boxes)

        # iterate over all objects

        for t in range(gt_boxes.size(0)):
            # compute the center of each gt box to determine which cell it falls on
            # assign it to a specific anchor by choosing max IoU

            gt_box_xywh = gt_boxes_xywh[t]
            gt_class = gt_classes[t]
            cell_idx_x, cell_idx_y = torch.floor(gt_box_xywh[:2])
            cell_idx = cell_idx_y * W + cell_idx_x
            cell_idx = cell_idx.long()

            # update box_target, box_mask
            overlaps_in_cell = overlaps[cell_idx, :, t]
            argmax_anchor_idx = torch.argmax(overlaps_in_cell)

            assigned_grid = all_grid_xywh.view(-1, num_anchors, 4)[cell_idx, argmax_anchor_idx, :].unsqueeze(0)
            gt_box = gt_box_xywh.unsqueeze(0)
            target_t = box_transform(assigned_grid, gt_box)
            if cfg.debug:
                print('assigned_grid, ', assigned_grid)
                print('gt: ', gt_box)
                print('target_t, ', target_t)
            box_target[b, cell_idx, argmax_anchor_idx, :] = target_t.unsqueeze(0)
            box_mask[b, cell_idx, argmax_anchor_idx, :] = 1

            # update cls_target, cls_mask
            class_target[b, cell_idx, argmax_anchor_idx, :] = gt_class
            class_mask[b, cell_idx, argmax_anchor_idx, :] = 1

            # update iou target and iou mask
            iou_target[b, cell_idx, argmax_anchor_idx, :] = max_iou[cell_idx, argmax_anchor_idx, :]
            if cfg.debug:
                print(max_iou[cell_idx, argmax_anchor_idx, :])
            iou_mask[b, cell_idx, argmax_anchor_idx, :] = cfg.object_scale

    return iou_target.view(bsize, -1, 1), \
        iou_mask.view(bsize, -1, 1), \
        box_target.view(bsize, -1, 4), \
        box_mask.view(bsize, -1, 1), \
        class_target.view(bsize, -1, 1).long(), \
        class_mask.view(bsize, -1, 1)


class WeightLoader(object):
    def __init__(self):
        super(WeightLoader, self).__init__()
        self.start = 0
        self.buf = None
        self.b = 'b'
        self.g = 'g'
        self.rm = 'rm'
        self.rv = 'rv'

    def load_conv_bn(self, conv_model, bn_model):

        num_w = conv_model.weight.numel()
        num_b = bn_model.bias.numel()

        bn_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        if bn_model.bias.data.shape == self.scratch.params['beta0'].shape:
            self.scratch.params['beta0'] = bn_model.bias.data
        elif bn_model.bias.data.shape == self.scratch.params['beta1'].shape:
            self.scratch.params['beta1'] = bn_model.bias.data
        elif bn_model.bias.data.shape == self.scratch.params['beta2'].shape:
            self.scratch.params['beta2'] = bn_model.bias.data
        elif bn_model.bias.data.shape == self.scratch.params['beta3'].shape:
            self.scratch.params['beta3'] = bn_model.bias.data
        elif bn_model.bias.data.shape == self.scratch.params['beta4'].shape:
            self.scratch.params['beta4'] = bn_model.bias.data
        elif bn_model.bias.data.shape == self.scratch.params['beta5'].shape:
            self.scratch.params['beta5'] = bn_model.bias.data
        elif (bn_model.bias.data.shape == self.scratch.params['beta6'].shape) and self.b == "b":
            self.scratch.params['beta6'] = bn_model.bias.data
            self.b = 'bb'
        elif (self.scratch.params['beta7'].shape == bn_model.bias.data.shape) and self.b == "bb":
            self.scratch.params['beta7'] = bn_model.bias.data

        self.start = self.start + num_b

        bn_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        if bn_model.weight.data.shape == self.scratch.params['gamma0'].shape:
            self.scratch.params['gamma0'] = bn_model.weight.data
        elif bn_model.weight.data.shape == self.scratch.params['gamma1'].shape:
            self.scratch.params['gamma1'] = bn_model.weight.data
        elif bn_model.weight.data.shape == self.scratch.params['gamma2'].shape:
            self.scratch.params['gamma2'] = bn_model.weight.data
        elif bn_model.weight.data.shape == self.scratch.params['gamma3'].shape:
            self.scratch.params['gamma3'] = bn_model.weight.data
        elif bn_model.weight.data.shape == self.scratch.params['gamma4'].shape:
            self.scratch.params['gamma4'] = bn_model.weight.data
        elif bn_model.weight.data.shape == self.scratch.params['gamma5'].shape:
            self.scratch.params['gamma5'] = bn_model.weight.data
        elif (bn_model.weight.shape == self.scratch.params['gamma6'].shape) and self.g == "g":
            self.scratch.params['gamma6'] = bn_model.weight.data
            self.g = 'gg'
        elif (self.scratch.params['gamma7'].shape == bn_model.weight.data.shape) and self.g == "gg":
            self.scratch.params['gamma7'] = bn_model.weight.data

        self.start = self.start + num_b

        bn_model.running_mean.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        if bn_model.running_mean.data.shape == self.scratch.bn_params[0]['running_mean'].shape:
            self.scratch.bn_params[0]['running_mean'] = bn_model.running_mean.data
        elif bn_model.running_mean.data.shape == self.scratch.bn_params[1]['running_mean'].shape:
            self.scratch.bn_params[1]['running_mean'] = bn_model.running_mean.data
        elif bn_model.running_mean.data.shape == self.scratch.bn_params[2]['running_mean'].shape:
            self.scratch.bn_params[2]['running_mean'] = bn_model.running_mean.data
        elif bn_model.running_mean.data.shape == self.scratch.bn_params[3]['running_mean'].shape:
            self.scratch.bn_params[3]['running_mean'] = bn_model.running_mean.data
        elif bn_model.running_mean.data.shape == self.scratch.bn_params[4]['running_mean'].shape:
            self.scratch.bn_params[4]['running_mean'] = bn_model.running_mean.data
        elif bn_model.running_mean.data.shape == self.scratch.bn_params[5]['running_mean'].shape:
            self.scratch.bn_params[5]['running_mean'] = bn_model.running_mean.data
        elif bn_model.running_mean.data.shape == self.scratch.bn_params[6]['running_mean'].shape and self.rm == "rm":
            self.scratch.bn_params[6]['running_mean'] = bn_model.running_mean.data
            self.rm = "rmrm"
        elif bn_model.running_mean.data.shape == self.scratch.bn_params[7]['running_mean'].shape and self.rm == "rmrm":
            self.scratch.bn_params[7]['running_mean'] = bn_model.running_mean.data

        self.start = self.start + num_b

        bn_model.running_var.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        if bn_model.running_var.data.shape == self.scratch.bn_params[0]['running_var'].shape:
            self.scratch.bn_params[0]['running_var'] = bn_model.running_var.data
        elif bn_model.running_var.data.shape == self.scratch.bn_params[1]['running_var'].shape:
            self.scratch.bn_params[1]['running_var'] = bn_model.running_var.data
        elif bn_model.running_var.data.shape == self.scratch.bn_params[2]['running_var'].shape:
            self.scratch.bn_params[2]['running_var'] = bn_model.running_var.data
        elif bn_model.running_var.data.shape == self.scratch.bn_params[3]['running_var'].shape:
            self.scratch.bn_params[3]['running_var'] = bn_model.running_var.data
        elif bn_model.running_var.data.shape == self.scratch.bn_params[4]['running_var'].shape:
            self.scratch.bn_params[4]['running_var'] = bn_model.running_var.data
        elif bn_model.running_var.data.shape == self.scratch.bn_params[5]['running_var'].shape:
            self.scratch.bn_params[5]['running_var'] = bn_model.running_var.data
        elif bn_model.running_var.data.shape == self.scratch.bn_params[6]['running_var'].shape and self.rv == "rv":
            self.scratch.bn_params[6]['running_var'] = bn_model.running_var.data
            self.rv = "rvrv"
        elif bn_model.running_var.data.shape == self.scratch.bn_params[7]['running_var'].shape and self.rv == "rvrv":
            self.scratch.bn_params[7]['running_var'] = bn_model.running_var.data

        self.start = self.start + num_b

        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))

        if conv_model.weight.data.shape == (16, 3, 3, 3):
            self.scratch.params['W0'] = conv_model.weight.data
        elif conv_model.weight.data.shape == (32, 16, 3, 3):
            self.scratch.params['W1'] = conv_model.weight.data
        elif conv_model.weight.data.shape == (64, 32, 3, 3):
            self.scratch.params['W2'] = conv_model.weight.data
        elif conv_model.weight.data.shape == (128, 64, 3, 3):
            self.scratch.params['W3'] = conv_model.weight.data
        elif conv_model.weight.data.shape == (256, 128, 3, 3):
            self.scratch.params['W4'] = conv_model.weight.data
        elif conv_model.weight.data.shape == (512, 256, 3, 3):
            self.scratch.params['W5'] = conv_model.weight.data
        elif conv_model.weight.data.shape == (1024, 512, 3, 3):
            self.scratch.params['W6'] = conv_model.weight.data
        elif conv_model.weight.data.shape == (1024, 1024, 3, 3):
            self.scratch.params['W7'] = conv_model.weight.data
        self.start = self.start + num_w

    def load_conv(self, conv_model):
        num_w = conv_model.weight.numel()
        num_b = conv_model.bias.numel()
        conv_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), conv_model.bias.size()))
        self.scratch.params['b8'] = conv_model.bias.data
        self.start = self.start + num_b
        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))
        self.scratch.params['W8'] = conv_model.weight.data
        self.start = self.start + num_w

    def dfs(self, m):
        children = list(m.children())
        for i, c in enumerate(children):
            if isinstance(c, torch.nn.Sequential):
                self.dfs(c)
            elif isinstance(c, torch.nn.Conv2d):
                if c.bias is not None:
                    self.load_conv(c)
                else:
                    self.load_conv_bn(c, children[i + 1])

    def load(self, model_to_load_weights_to, model, weights_file):
        self.scratch = model_to_load_weights_to
        self.start = 0
        fp = open(weights_file, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.buf = np.fromfile(fp, dtype=np.float32)
        fp.close()
        size = self.buf.size
        self.dfs(model)
        # make sure the loaded weight is right
        assert size == self.start
        return self.scratch


_Load_Weights = True
if _Load_Weights:
    class Yolov2(nn.Module):

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
            self.bn1 = nn.BatchNorm2d(16)

            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(32)

            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(64)

            self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn4 = nn.BatchNorm2d(128)

            self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn5 = nn.BatchNorm2d(256)

            self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn6 = nn.BatchNorm2d(512)

            self.conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn7 = nn.BatchNorm2d(1024)

            self.conv8 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn8 = nn.BatchNorm2d(1024)

            self.conv9 = nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1)

        def forward(self, x, gt_boxes=None, gt_classes=None, num_boxes=None, training=False):

            x = self.maxpool(self.lrelu(self.bn1(self.conv1(x))))
            x = self.maxpool(self.lrelu(self.bn2(self.conv2(x))))
            x = self.maxpool(self.lrelu(self.bn3(self.conv3(x))))
            x = self.maxpool(self.lrelu(self.bn4(self.conv4(x))))
            x = self.maxpool(self.lrelu(self.bn5(self.conv5(x))))
            x = self.lrelu(self.bn6(self.conv6(x)))
            # x = F.pad(x, (0, 1, 0, 1))
            # x = self.slowpool(x)
            x = self.lrelu(self.bn7(self.conv7(x)))
            x = self.lrelu(self.bn8(self.conv8(x)))
            out = self.conv9(x)

            # out -- tensor of shape (B, num_anchors * (5 + num_classes), H, W)
            bsize, _, h, w = out.size()

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

                target_variable = [v for v in target_data]
                box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)

                return box_loss, iou_loss, class_loss

            return delta_pred, conf_pred, class_pred

pytorch_model = DeepConvNet(input_dims=(3, 416, 416),
                            num_filters=[16, 32, 64, 128, 256, 512, 1024, 1024],
                            max_pools=[0, 1, 2, 3, 4],
                            weight_scale='kaiming',
                            batchnorm=True,
                            dtype=torch.float32, device='cpu')

# model = Yolov2()
# weightloader = WeightLoader()
# pytorch_model = weightloader.load(pytorch_model, model, "data/pretrained/yolov2-tiny-voc.weights")
#
# if __name__ == '__main__':
#
#     Weight, Bias, Beta, Gamma, Running_Mean, Running_Var = pytorch_model.Training_Parameters()
#     # print(type(Weight[0]))
#     # print(type(Weight[1]))
#     # print(Bias)
#     print(Running_Mean[0])
#     print(Running_Var[0])
