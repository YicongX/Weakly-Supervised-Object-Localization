import copy
import os
import random
import time
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# given bounding boxes and corresponding scores, perform non max suppression
def nms(bounding_boxes, confidence_score, threshold=0.4):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """
    matrix = np.hstack((bounding_boxes,confidence_score.reshape(-1,1)))
    x1 = matrix[:, 0]
    y1 = matrix[:, 1]
    x2 = matrix[:, 2]
    y2 = matrix[:, 3]
    scores = matrix[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    good_bx = []
    while order.size > 0:
        i = order[0]
        good_bx.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return good_bx
    



#calculate the intersection over union of two boxes
def get_iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """
    topl = np.maximum(box1[:, None, :2], box2[:, :2])
    botr = np.minimum(box1[:, None, 2:], box2[:, 2:])

    #intersection
    area_iter = np.prod(botr - topl, axis=2) * (topl < botr).all(axis=2)
    #box1
    area_b1 = np.prod(box1[:, 2:] - box1[:, :2], axis=1)
    #box2
    area_b2 = np.prod(box2[:, 2:] - box2[:, :2], axis=1)
    iou = area_iter / (area_b1[:, None] + area_b2 - area_iter)
    return iou



def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image


def get_box_data(classes, bbox_coordinates):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
            "position": {
                "minX": bbox_coordinates[i][0],
                "minY": bbox_coordinates[i][1],
                "maxX": bbox_coordinates[i][2],
                "maxY": bbox_coordinates[i][3],
            },
            "class_id": classes[i],
        } for i in range(len(classes))
        ]

    return box_list

def get_bbox(classes, score, bbox_coordinates):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
            "position": {
                "minX": float(bbox_coordinates[i][0]),
                "minY": float(bbox_coordinates[i][1]),
                "maxX": float(bbox_coordinates[i][2]),
                "maxY": float(bbox_coordinates[i][3]),
            },
            "class_id": classes[i],
            "scores": {"score":float(score[i])},
        } for i in range(len(classes))
        ]

    return box_list
