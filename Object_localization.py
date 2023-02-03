import argparse
import os
import shutil
import time

import sklearn
import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import fbeta_score
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.data.sampler import SequentialSampler
import torchvision.datasets as datasets
import torchvision.models as models
import wandb
import matplotlib.pyplot as plt

from AlexNet import localizer_alexnet, localizer_alexnet_robust
from voc_dataset import *
from utils import *

USE_WANDB = False  # use flags, wandb is not convenient for debugging
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet_robust')#localizer_alexnet_robust
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=45,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=32,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.01,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_false',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')
parser.add_argument(
    '--num_heatmap', default=2, type=int,
    help='Number of images to display in every batch')

best_prec1 = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache() 

def heatmap_pro(heatmap,size=512):
    cmap = plt.get_cmap('jet')
    heatmap = heatmap[0]
    upsample = transforms.Compose([transforms.ToPILImage(),transforms.Resize((size,size))])
    scaler = MinMaxScaler()
    scaler.fit(heatmap)
    heatmap = scaler.transform(heatmap)
    heatmap = upsample(torch.Tensor(heatmap).unsqueeze(0))
    heatmap= np.uint8(cmap(np.array(heatmap))*255)
    return heatmap

def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), 
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    dataset = VOCDataset('trainval', top_n=None)
    val_split = .2
    data_size = len(dataset)
    data_idx = list(range(data_size))
    split = int(np.floor(val_split * data_size))
    train_idx, val_idx = data_idx[split:], data_idx[:split]


    
    
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_idx,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_idx,
        drop_last=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    USE_WANDB = True
    if USE_WANDB:
        wandb.init(project="vlr-hw1", reinit=True)

    m1_val = []
    m2_val = []
    m1_train = []
    m2_train = []
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch

        print('Epoch:', epoch,'LR:', lr_scheduler.get_last_lr())
        train(train_loader, model, criterion, optimizer, epoch,m1_train,m2_train)
        lr_scheduler.step()

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, epoch, m1_val, m2_val)

            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
    plt.plot(m1_train)
    plt.ylabel('metric 1')
    plt.ylim([0, 1.3])
    wandb.log({"Train_m1": plt})

    plt.plot(m2_train)
    plt.ylabel('metric 2')
    plt.ylim([0, 1.3])
    wandb.log({"Train_m2": plt})

    plt.plot(m1_val)
    plt.ylabel('metric 1')
    wandb.log({"Val_m1": plt})

    plt.plot(m2_val)
    plt.ylabel('metric 2')
    wandb.log({"Val_m2": plt})
        

def train(train_loader, model, criterion, optimizer, epoch,m1_train,m2_train):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (data) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Convert inputs to cuda if training on GPU
        
        target = data['label'].type(torch.cuda.FloatTensor).to(device)
        input_img = data['image'].type(torch.cuda.FloatTensor).to(device)
        
        imoutput = model(input_img)

        output = F.max_pool2d(imoutput, kernel_size = (imoutput.shape[2:]) )
        output = torch.squeeze(output)

        loss = criterion(output,target)
        

        # measure metrics and record loss
        m1 = metric1(output.data, target)
        m2 = metric2(output.data, target)
        losses.update(loss.item(), input_img.size(0))
        avg_m1.update(m1[0],input_img.size(0))
        avg_m2.update(m2[0],input_img.size(0))
        m1_train.append(m1)
        m2_train.append(m2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        wandb.log({'Train loss': loss})
        if i == len(train_loader)-1:
            gt_data = data

        # Visualize Heat Map
    if epoch == 0 or (epoch+1) % 15 == 0:
        for img_num in range(args.num_heatmap):
            gt_imgs = tensor_to_PIL(gt_data['image'][img_num])
            heatmap = imoutput[img_num,:,:,:].data.cpu().numpy()
            gt_label = gt_data['label'][img_num].cpu().numpy()
            gt_label = np.nonzero(gt_label)[0].astype(int)
            heatmap = heatmap[gt_label]
            heatmap_out = heatmap_pro(heatmap)
            images = [gt_imgs,heatmap_out]
            wandb.log({"Train: Epoch%d: Img%d" %(epoch,img_num):[wandb.Image(image) for image in images]})

        # End of train()


def validate(val_loader, model, criterion, epoch, m1_val, m2_val,):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()
    

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data) in enumerate(val_loader):
        # Convert inputs to cuda if training on GPU
        input_img = data['image'].to(device)
        target = data['label'].to(device)

        imoutput = model(input_img)

        output = F.max_pool2d(imoutput, kernel_size = (imoutput.shape[2:]) )
        output = torch.squeeze(output)

        loss = criterion(output,target)

        # measure metrics and record loss
        m1 = metric1(output.data, target)
        m2 = metric2(output.data, target)
        losses.update(loss.item(), input_img.size(0))
        avg_m1.update(m1[0],input_img.size(0))
        avg_m2.update(m2[0],input_img.size(0))
        if epoch == 0 or (epoch + 1) % 2 == 0:
            m1_val.append(m1)
            m2_val.append(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        wandb.log({'Validation loss': loss})
        if i == len(val_loader)-1:
            gt_data = data

        # Visualize Heat Map
    if epoch+1 == args.epochs:
        for img_num in range(3):
            rand_idx = [5,10,15]
            gt_imgs = tensor_to_PIL(gt_data['image'][rand_idx[img_num]])
            heatmap = imoutput[rand_idx[img_num],:,:,:].data.cpu().numpy()
            gt_label = gt_data['label'][rand_idx[img_num]].cpu().numpy()
            gt_label = np.nonzero(gt_label)[0].astype(int)
            heatmap = heatmap[gt_label]
            heatmap_out = heatmap_pro(heatmap)
            images = [gt_imgs,heatmap_out]
            wandb.log({"Validation: Epoch%d: Img%d" %(epoch,img_num):[wandb.Image(image) for image in images]})
    

    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def metric1(output, target, threshold = 0.5):
    num_cls = target.shape[1]
    sol = []
    output = torch.sigmoid(output)
    output = output > threshold
    for img in range(num_cls):
        true_cls = target[:, img].cpu().numpy().astype('float32')
        pred_cls = output[:, img].cpu().numpy().astype('float32')
        if np.count_nonzero(true_cls) == 0:
            continue
        else:
            prec = sklearn.metrics.average_precision_score(true_cls, pred_cls,average='micro')
        sol.append(prec)
    return [np.mean(sol)]

    


def metric2(output, target,threshold = 0.5):
    output = torch.sigmoid(output)
    output = output > threshold
    sol = sklearn.metrics.recall_score(np.int32(target.cpu().numpy()), np.int32(output.cpu().numpy()), average='micro')
    return [sol]

if __name__ == '__main__':
    main()
