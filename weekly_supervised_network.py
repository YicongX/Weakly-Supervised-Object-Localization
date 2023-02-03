from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import argparse
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl
import time
import six
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt

from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import *
from PIL import Image, ImageDraw



# hyper-parameters
# ------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    '--lr',
    default=0.00009,
    type=float,
    #description='Learning rate'
)
parser.add_argument(
    '--batch',
    default=1,
    type=int,
    #description='Flag to enable visualization'
)
parser.add_argument(
    '--lr-decay-steps',
    default=150000,
    type=int,
    #description='Interval at which the lr is decayed'
)
parser.add_argument(
    '--lr-decay',
    default=0.1,
    type=float,
    #description='Decay rate of lr'
)
parser.add_argument(
    '--momentum',
    default=0.9,
    type=float,
    #description='Momentum of optimizer'
)
parser.add_argument(
    '--weight-decay',
    default=0.00005,
    type=float,
    #description='Weight decay'
)
parser.add_argument(
    '--epochs',
    default=5,
    type=int,
    #description='Number of epochs'
)
parser.add_argument(
    '--val-interval',
    default=5000,
    type=int,
    #description='Interval at which to perform validation'
)
parser.add_argument(
    '--disp-interval',
    default=10,
    type=int,
    #description='Interval at which to perform visualization'
)
parser.add_argument(
    '--use-wandb',
    default=False,
    type=bool,
    #description='Flag to enable visualization'
)
parser.add_argument(
    '--top_n',
    default=300,
    type=int,
    #description='Flag to enable visualization'
)
parser.add_argument(
    '--val_interval',
    default=50,
    type=int,
    #description='Flag to enable visualization'
)
# ------------

# Set random seed
rand_seed = 1024
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

# Set output directory
output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def calculate_map(all_pred_boxes,all_pred_labels,all_pred_scores,all_gt_boxes,all_gt_labels,iou_thresh = 0.5):
    """
    Calculate the mAP for classification.
    """

    all_pred_boxes = iter(all_pred_boxes)
    all_pred_labels = iter(all_pred_labels)
    all_pred_scores = iter(all_pred_scores)
    all_gt_boxes = iter(all_gt_boxes)
    all_gt_labels = iter(all_gt_labels)
    

    n_pos = defaultdict(int)
    score = defaultdict(list)
    correct = defaultdict(list)

    for pred_box, pred_label, pred_score, gt_bbox, gt_label in \
        six.moves.zip( all_pred_boxes, all_pred_labels, all_pred_scores,all_gt_boxes, all_gt_labels):

        gt_diff = np.zeros(gt_bbox.shape[0], dtype=bool)

        for idx in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_idx = pred_label == idx
            pred_box_idx = pred_box[pred_mask_idx]
            pred_score_idx = pred_score[pred_mask_idx]                  # Get box and score with each unique label
            
            order = pred_score_idx.argsort()[::-1]
            pred_box_idx = pred_box_idx[order]                          # sort box based on box score
            pred_score_idx = pred_score_idx[order]

            gt_mask_idx = gt_label == idx
            gt_box_idx = gt_bbox[gt_mask_idx]                           # find ground truth label that appeared in prediction
            gt_diff_idx = gt_diff[gt_mask_idx]

            n_pos[idx] += np.logical_not(gt_diff_idx).sum()
            score[idx].extend(pred_score_idx)

            if len(pred_box_idx) == 0:
                continue
            if len(gt_box_idx) == 0:
                correct[idx].extend((0,) * pred_box_idx.shape[0])       # if no ground choose box, add zeros to correct
                continue

            
            pred_box_idx = pred_box_idx.copy()
            pred_box_idx[:, 2:] += 1
            gt_box_idx = gt_box_idx.copy()
            gt_box_idx[:, 2:] += 1

            iou = get_iou(pred_box_idx, gt_box_idx)
            gt_index = iou.argmax(axis=1)
            
            gt_index[iou.max(axis=1) < iou_thresh] = -1                 #-1 if no groud truth box associated
            del iou

            selec = np.zeros(gt_box_idx.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_diff_idx[gt_idx]:
                        correct[idx].append(-1)
                    else:
                        if not selec[gt_idx]:
                            correct[idx].append(1)
                        else:
                            correct[idx].append(0)
                    selec[gt_idx] = True
                else:
                    correct[idx].append(0)

    n_class = max(n_pos.keys()) + 1
    prec = [None] * n_class
    rec = [None] * n_class

    for idx in n_pos.keys():
        score_idx = np.array(score[idx])
        correct_idx = np.array(correct[idx], dtype=np.int8)

        order = score_idx.argsort()[::-1]
        correct_idx = correct_idx[order]

        tp = np.cumsum(correct_idx == 1)                    
        fp = np.cumsum(correct_idx == 0)

        prec[idx] = tp / (fp + tp)                                   # If an element of fp + tp is 0,precision is nan.
        if n_pos[idx] > 0:
            rec[idx] = tp / n_pos[idx]                               # If no positive ground truth, recall is None.

    n_class = len(prec)
    ap = np.empty(n_class)
    for idx in six.moves.range(n_class):
        if prec[idx] is None or rec[idx] is None:                    # ap is none when either precison or recall is none
            ap[idx] = np.nan
            continue

        precision = np.concatenate(([0], np.nan_to_num(prec[idx]), [0]))
        recall = np.concatenate(([0], rec[idx], [1]))

        precision = np.maximum.accumulate(precision[::-1])[::-1]

        chg_pt = np.where(recall[1:] != recall[:-1])[0]                       # look for points where recall changes value to calculate area under precision recall curve

        ap[idx] = np.sum((recall[chg_pt + 1] - recall[chg_pt]) * precision[chg_pt + 1])      # sum delta(recall)*precision

    return {'ap': ap, 'map': np.nanmean(ap)}


def test_model(model, val_loader, visualize, dataset, thresh=0.05):
    """
    Tests the networks and visualizes the detections
    :param thresh: Confidence threshold
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    class_id_to_label = dict(enumerate(dataset.CLASS_NAMES))
    with torch.no_grad():
        all_gt_boxes = []
        all_gt_labels = []
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_labels = []

        for iter, data in enumerate(val_loader):

            # one batch = data for one image
            
            image = data['image']
            target = data['label']
            #wgt = data['wgt']
            rois = data['rois']
            gt_boxes = data['gt_boxes']
            gt_class_list = data['gt_classes']
            for i in range (len(gt_boxes)):
                gt_boxes[i] = torch.cat(gt_boxes[i],axis = 0)
            gt_boxes = [data.numpy() for data in gt_boxes]
            gt_boxes = np.array(gt_boxes).reshape(-1,4)
            gt_class_list = [data.numpy() for data in gt_class_list]
            gt_class_list = np.array(gt_class_list).reshape(-1)
                
            image = image.type(torch.cuda.FloatTensor).cuda()
            img_size = image.shape[2]
            rois_real = img_size * rois.type(torch.cuda.FloatTensor).cuda()
            gt_vec = target.cuda()

            cls_prob = model(image,rois_real,gt_vec)
            loss = model.loss
            batch_box_score = model.box_score

            batch_box_score = batch_box_score.data.cpu().numpy().squeeze()
            scores = cls_prob.data.cpu().numpy()
            rois = rois.numpy().squeeze()
            cls_score = []
            batch_gt_labels = gt_class_list
            batch_gt_boxes = gt_boxes
            batch_pred_label = []
            batch_pred_boxes = []
            batch_pred_scores = []
            thresh_box = 0.05
            for class_num in range(20):
                # get valid rois and cls_scores based on thresh
                if scores[class_num] > thresh and len(batch_gt_boxes)!=0:
                    # if iter == 1000 or iter == 50 or iter == 800:
                    #     print(batch_box_score[:,class_num])
                    box_candi = rois[np.where(batch_box_score[:,class_num]>thresh_box)]
                    box_score = batch_box_score[np.where(batch_box_score[:,class_num]>thresh_box),class_num].reshape(-1,1)
                # use NMS to get boxes and scores
                    good_idx = nms(box_candi, box_score)
                    batch_pred_boxes.append(box_candi[good_idx])
                    batch_pred_scores.append(box_score[good_idx])
                    for i in range(len(good_idx)):
                        batch_pred_label.append(class_num)



            if len(batch_gt_boxes)!=0:
                all_gt_boxes.append(batch_gt_boxes)
                all_gt_labels.append(batch_gt_labels)
 
                if len(batch_pred_boxes) != 0:
                    all_pred_boxes.append(np.concatenate(batch_pred_boxes, axis=0))
                    all_pred_scores.append(np.concatenate(batch_pred_scores, axis=0).reshape(-1))
                    all_pred_labels.append(batch_pred_label)
                else:
                    all_pred_boxes.append(np.empty((1,4)))
                    all_pred_scores.append(np.empty((1)))
                    all_pred_labels.append(np.empty((1)))
                    
            losses.update(loss.item(), image.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if visualize == True and 20 < (len(val_loader) - 1 - iter) < 31 and len(batch_gt_boxes)!=0:
                original_image = tensor_to_PIL(image[0])
                img = wandb.Image(original_image, boxes={
                    "predictions": {
                        "box_data": get_bbox(all_pred_labels[iter],all_pred_scores[iter], all_pred_boxes[iter]),
                        "class_labels": class_id_to_label,
                    },
                })
                #wandb.log({"Validation%d" %iter:img}) 

            result = calculate_map(all_pred_boxes,all_pred_labels,all_pred_scores,all_gt_boxes,all_gt_labels)

            if iter == len(val_loader)-1:
                print('Iteration: [{0}/{1}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'mAP [{2}]'.format(
                        iter,
                        len(val_loader),
                        result['map'],
                        batch_time=batch_time,
                        loss=losses))
                #wandb.log({'Validation loss': loss})
    return result
            


def train_model(model, train_loader, val_loader, optimizer, args,dataset):
    """
    Trains the network, runs evaluation and visualizes the detections
    """
    # Initialize training variables
    train_loss = 0
    step_cnt = 0
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    visualize = False
    mAP = []
    class_ap = []
    for epoch in range(args.epochs):
        if epoch % 2 == 0:
            visualize = True
        for iter, data in enumerate(train_loader):

            # one batch = data for one image
            image = data['image']
            target = data['label']
            #wgt = data['wgt']
            rois = data['rois']
            #gt_boxes = data['gt_boxes']
            gt_class_list = data['gt_classes']

            # Convert inputs to cuda if training on GPU
            image = image.type(torch.cuda.FloatTensor).cuda()
            img_size = image.shape[2]
            rois *= img_size
            rois = rois.type(torch.cuda.FloatTensor).cuda()
            
            gt_vec = target.cuda()
            model(image,rois,gt_vec)


            # backward pass and update
            loss = model.loss
            train_loss += loss.item()
            step_cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), image.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if iter % 500 == 0:
                print('Iteration: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                      epoch,
                      iter,
                      len(train_loader),
                      batch_time=batch_time,
                      loss=losses))
                #wandb.log({'Train loss': loss})

            if epoch % 2 == 0 and iter % 500 == 0 and iter != 0: #wandb.log({'Train loss': loss})
                model.eval()
                ap = test_model(model, val_loader,visualize,dataset)
                print("AP ", ap)
                model.train()
                visualize = False
                mAP.append(ap['map'])
                class_ap.append(ap['ap'])
                

    # Plot class-wise APs
    class_ap = np.array(class_ap)
    for i in range(5):
        plt.plot(class_ap[:,i])
        plt.ylabel('Class %d: AP'%i)
        #wandb.log({"Class %d: AP"%i: plt})

    plt.plot(mAP)
    plt.ylabel('mAP')
    #plt.ylim([0, 1.3])
    #wandb.log({"mAP": plt})


def main():
    """
    Creates dataloaders, network, and calls the trainer
    """
    args = parser.parse_args()
    torch.cuda.empty_cache()

    # Initialize wandb logger
    dataset = VOCDataset('trainval', top_n=args.top_n)
    val_split = .2
    data_size = len(dataset)
    data_idx = list(range(data_size))
    split = int(np.floor(val_split * data_size))
    train_idx, val_idx = data_idx[split:], data_idx[:split]
    
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=train_idx,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=val_idx,
        drop_last=True)
    
    USE_WANDB = False
    if USE_WANDB:
        wandb.init(project="vlr-hw1", reinit=True)

    # Create network and initialize
    net = WSDDN(classes=dataset.CLASS_NAMES)
    #print(net)

    if os.path.exists('pretrained_alexnet.pkl'):
        pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
    else:
        pret_net = model_zoo.load_url(
            'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        pkl.dump(pret_net,
        open('pretrained_alexnet.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
    own_state = net.state_dict()

    for name, param in pret_net.items():
        #print(name)
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
            #print('Copied {}'.format(name))
        except:
            #print('Did not find {}'.format(name))
            continue

    # Move model to GPU and set train mode
    net.load_state_dict(own_state)
    net.cuda()
    net.train()

    for param in net.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(net.parameters(), 
                                args.lr,
                                weight_decay=args.weight_decay)

    # Training
    train_model(net, train_loader, val_loader, optimizer, args, dataset)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
if __name__ == '__main__':
    main()