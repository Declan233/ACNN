import torch
import torch.nn as nn
from utils import compute_iou, compute_intersection

# This new loss function is based on https://github.com/ultralytics/yolov3/blob/master/utils/loss.py


def compute_loss(predictions:torch.Tensor, targets:torch.Tensor, stride:int):
    _alpha = 0.5

    if targets is None:
        return 0
    else:
        format_output = build_target(pred_AOI = predictions[..., :2], # sigmoid(ps_t), pr_t
                                     target = targets, 
                                     scale = stride)

    pred_conf = predictions[..., 2]
    pred_cls = predictions[..., 3:]

    iou_scores = format_output['iou_scores']
    leak_mask = format_output['leak_mask']
    noleak_mask = format_output['noleak_mask']
    tcls = format_output['tcls']
    tconf = format_output['leak_mask'].float()

    mes_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    loss_conf_obj = mes_loss(pred_conf[leak_mask], tconf[leak_mask])
    loss_conf_noobj = mes_loss(pred_conf[noleak_mask], tconf[noleak_mask])
    loss_cls = ce_loss(pred_cls[leak_mask], tcls[leak_mask])
    loss_iou = (1.0-iou_scores[leak_mask]).mean()

    loss = 10*(_alpha*loss_conf_obj + (1-_alpha)*loss_conf_noobj) + loss_iou + loss_cls 

    loss_dict = {}
    loss_dict['loss_conf_leak'] = loss_conf_obj.item()
    loss_dict['loss_conf_noleak'] = loss_conf_noobj.item()
    loss_dict['loss_iou'] = loss_iou.item()
    loss_dict['loss_cls'] = loss_cls.item()
    loss_dict['loss'] = loss.item()
    # loss_dict['cls_acc'] = (100 * class_mask[leak_mask].mean()).item()
    return loss, loss_dict


def build_target(pred_AOI, target, scale):
    ''' Generating mask for selecting valid windows.

    Input:
    - pred_AOI: bs*nl*2 (sigmoid(ps_t), pr_t)
    - target: bs*3 (ps, pr, cls_target)
    - winsow: window size
    - scale: scale factor, a.k.a. window stride.

    Output:
    - leak_mask
    - noleak_mask
    - iou_scores: the iou(pred_AOI, target_AOI) conditioned on leak_mask
    - tcls: target class conditioned on leak_mask
    '''
    # ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    BoolTensor = torch.cuda.BoolTensor if pred_AOI.is_cuda else torch.BoolTensor
    LongTensor = torch.cuda.LongTensor if pred_AOI.is_cuda else torch.LongTensor
    FloatTensor = torch.cuda.FloatTensor if pred_AOI.is_cuda else torch.FloatTensor

    bs, nl, _ = pred_AOI.shape

    leak_mask   = BoolTensor(bs, nl).fill_(0)
    noleak_mask = BoolTensor(bs, nl).fill_(1)
    iou_scores = FloatTensor(bs, nl).fill_(0)

    tcls = LongTensor(bs, nl).fill_(0)

    # Convert to relative coordination representation
    ps = target[:, 0] / scale # target AOI center
    pr = target[:, 1] / scale # target AOI range
    target_label = target[:, 2].long() #cls_target
    li = ps.long() # window index

    b = torch.arange(bs).view(bs,).long()

    # set leak mask
    leak_mask[b, li] = True
    # Set noleak mask
    noleak_mask[b, li] = False
    
    # convert to net output representation
    tcls[b, li] = target_label # label

    # target AOI in relative representation: (ps/scale-li, pr/scale)
    target_AOI = torch.cat(((ps-li).view(-1,1), pr.view(-1,1)), 1) #(bs, 2)

    pred_bbox = pred_AOI[b, li]
    pred_bbox[:, 1] = torch.exp(pred_bbox[:, 1]) 
    iou_scores[b, li] = compute_iou(target_AOI.t(), pred_bbox.t()).float()

    return {'iou_scores': iou_scores,
            'tcls': tcls,
            'leak_mask': leak_mask, 
            'noleak_mask': noleak_mask}

