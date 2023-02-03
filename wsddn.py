import numpy as np
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.ops import roi_pool, roi_align


class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=None):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)

        #Define the WSDDN model
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),  
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),  
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  
            nn.ReLU(),
        )
        self.roi_pool = roi_pool
        self.classifier = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(),
            #nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            )

        self.score_fc = nn.Linear(4096, 20)

        self.bbox_fc = nn.Linear(4096, 20)
        # loss
        self.cross_entropy = None
        self.box_score = None

    @property
    def loss(self):
        return self.cross_entropy

    def box_score(self):
        return self.box_score

    def forward(self,
                image,
                rois,
                gt_vec,
                ):

        # compute cls_prob which are N_roi X 20 scores
        rois = [roi.type(torch.float) for roi in rois]
        out_features = self.features(image)
        
        roi_features = self.roi_pool(out_features, rois, output_size = [6,6],spatial_scale = 31/512)
        
        roi_features = roi_features.view(roi_features.shape[0], -1)
        
        cls_output = self.classifier(roi_features)

        cls_soft = F.softmax(self.score_fc(cls_output),dim = 1)

        box_soft = F.softmax(self.bbox_fc(cls_output),dim = 0)

        self.box_score = box_soft

        cls_prob = torch.sum(cls_soft*box_soft, dim = 0).view(self.n_classes,-1)

        if self.training:
            label_vec = gt_vec.view(self.n_classes, -1)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)
        return cls_prob

    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        cls_prob = torch.clamp(cls_prob, 0, 1)
        loss = F.binary_cross_entropy(cls_prob, label_vec,reduction="sum")
        return loss
