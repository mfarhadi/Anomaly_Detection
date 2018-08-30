# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
import skimage.measure

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="models")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true',default="cuda")
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=17, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def var_ghamar(avg,cur):
    if(avg[4]==3):
        return cur,cur,cur,cur,4,0
    else:
        difference=abs(avg[0]-cur)
        detected=0
        N = avg[4] + 1
        if (difference>(np.sqrt(avg[1])*3)):
            detected=1
            if (N>8):
                N=int(N/2)

        min=avg[2]
        if(cur<avg[2]):
            min=cur
        max=avg[3]
        if(cur>avg[3]):
            max=cur
        mean = ((avg[0]*(N-1))+(cur))/(N)

        variance=((((N-2)*avg[1])+ (cur-mean)*(cur-avg[0]))/(N-1))
        #print (mean,variance,min,max,N,detected)
        return mean,variance,min,max,N,detected

def draw_flow(im,flow,step=24):
    h,w = im.shape[:2]

    y,x = np.mgrid[int(step/2):h:step,int(step)/2:w:step].reshape(2,-1)
    fx,fy = flow[y,x].T

    # create line endpoints
    lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)

    # create image and draw
    vis =im #cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
    return vis
def destribution_creator(frame2,prvs):
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((next.shape[0], next.shape[1]))
    anomaly_map=np.zeros((next.shape[0], next.shape[1],3))
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    Pool1_flow = skimage.measure.block_reduce(flow, (2, 2, 1), np.max)
    Pool2_flow = skimage.measure.block_reduce(Pool1_flow, (2, 2, 1), np.average)
    limit=Pool2_flow.shape
    flow2 = Pool2_flow
    itemindex = np.where(flow2 > 1)
    itemindex = itemindex + np.where(flow2 < -1)
    size = itemindex[1].shape
    for i in range(size[0]):
        index_1 = itemindex[0][i]
        index_2 = itemindex[1][i]
        index_3 = itemindex[2][i]
        destribution_motion[index_1][index_2][index_3][0], destribution_motion[index_1][index_2][index_3][1], \
        destribution_motion[index_1][index_2][index_3][2], destribution_motion[index_1][index_2][index_3][3], \
        destribution_motion[index_1][index_2][index_3][4], block_anomaly = var_ghamar(destribution_motion[index_1][index_2][index_3],
                                                                               flow2[index_1][index_2][index_3])
        mask[index_1][index_2]=1
        if (block_anomaly == 1):
            if(index_1<(limit[0]-1)):
                for jj in range(index_1 * 4, (index_1 * 4) + 4):
                    if (index_2 < (limit[1] - 1)):
                        for jjj in range(index_2 * 4, (index_2 * 4) +4):
                            anomaly_map[jj][jjj][2] = 254
    return flow,frame2,mask,anomaly_map


def object_detection(im,result):
    blobs, im_scales = _get_image_blob(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
    im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
    gt_boxes.data.resize_(1, 1, 5).zero_()
    num_boxes.data.resize_(1).zero_()

    # pdb.set_trace()
    det_tic = time.time()
    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    result_box=[]
    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    det_toc = time.time()
    detect_time = det_toc - det_tic
    misc_tic = time.time()
    if vis:
        im2show = np.copy(result)

    for j in xrange(1, len(pascal_classes)):
        inds = torch.nonzero(scores[:, j] > thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
            else:
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            result_box.append([cls_dets.cpu(),j])
            if vis:
                im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)

    misc_toc = time.time()
    nms_time = misc_toc - misc_tic

    # sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
    # .format(i + 1, num_images, detect_time, nms_time))
    # sys.stdout.flush()
    return result_box,im2show
def combined(previous,rows,cols,destribution_motion,destribution_object,vid,test,AnomalyBoundary):
    counter = 0
    FrameCounter=0
    TrueAnomaly_counter=0
    FalseAnomaly_counter = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # RECOURD RESULT
    AnomalyStart=None
    AnomalyStop=None
    AnomalyIndex=0
    AnomalyFrame=0
    TotalAnomalyFrames=0
    stack=[]
    while (True):
        counter += 1

        ret, frame = cap.read()

        if (ret == False):
            break
        #frame = cv2.resize(frame, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_CUBIC)
        im = frame
        im2 = im
        rows, cols = frame.shape[0], frame.shape[1]
        flow, frame2, mask,anomaly_map = destribution_creator(frame, previous)
        # delay on previus and curent frame
        if(counter<4):
            previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            stack.append(previous)
        else:
            previous=stack.pop()
            stack.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        predicted_boxes, im2show = object_detection(im, frame2)
        # print(predicted_boxes)
        destribution_object_tmp = np.zeros((rows, cols, 22))

        for label in predicted_boxes:
            for position in label[0]:
                if (position[4] > 0.95):
                    # print('mehdi',label[1],int(position[0]),int(position[1]),int(position[2]),int(position[3]))
                    # mask2[int(position[1]):int(position[3]),int(position[0]):int(position[2]),2]=np.add(mask2[int(position[1]):int(position[3]),int(position[0]):int(position[2]),2],255)
                    destribution_object_tmp[int(position[1]):int(position[3]), int(position[0]):int(position[2]),
                    int(label[1])] = np.add(
                        destribution_object_tmp[int(position[1]):int(position[3]), int(position[0]):int(position[2]),
                        int(label[1])], 1)
                    destribution_object_tmp[int(position[1]):int(position[3]), int(position[0]):int(position[2]),
                    21] = np.add(
                        destribution_object_tmp[int(position[1]):int(position[3]), int(position[0]):int(position[2]),
                        21], 1)
                    # print('ame mehdi',jj)

                    # index2=np.where(destribution_object_tmp[:,:,21] > 1)
                    # print('tmp',index2[1].shape)
                    # index2=np.where(destribution_object[:,:,21] > 1)
                    # print('prev',index2[1].shape)
        destribution_object = np.add(destribution_object, destribution_object_tmp)
        # index2=np.where(destribution_object[:,:,21] > 1)
        # print('new',index2[1].shape)
        mask2 = np.zeros((rows, cols, 3))
        mask3 = np.zeros((rows, cols, 3))
        mask4 = np.zeros((rows, cols, 3))
        object_probability = np.zeros((rows, cols, 21))
        mask2[:, :, 2] = np.divide(destribution_object[:, :, 15], destribution_object[:, :, 21])
        for ame_mehdi in range(0, 21):
            object_probability[:, :, ame_mehdi] = np.divide(destribution_object[:, :, ame_mehdi],
                                                            destribution_object[:, :, 21])
        object_probability = np.multiply(object_probability, destribution_object_tmp[:, :, 0:21])
        itemindex = np.where(object_probability > 0.4) # high probability are removing
        size = itemindex[1].shape
        for i in range(size[0]):
            index_1 = itemindex[0][i]
            index_2 = itemindex[1][i]
            index_3 = itemindex[2][i]
            object_probability[index_1][index_2][index_3] = 0
        mask3[:, :, 0:2] = destribution_motion[:, :, :, 0]  # mean distribuation ro show mikonam
        itemindex = np.where(object_probability > 0)
        size = itemindex[1].shape
        for i in range(size[0]):
            index_1 = itemindex[0][i]
            index_2 = itemindex[1][i]
            index_3 = itemindex[2][i]
            mask4[index_1][index_2][1] = 254


        mask3.astype('uint8')
        crop_size=mask4.shape
        mask3 = cv2.resize(mask3, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        mask3=mask3[0:crop_size[0],0:crop_size[1],:]   #Motion Destribution
        mask3 = np.multiply(mask3, 120)
        cv2.putText(mask3, 'Motion Destribution', (0,0), cv2.FONT_HERSHEY_PLAIN,1, (255, 255, 255), thickness=1)
        mask4=mask4+anomaly_map                        #ANomaly_MAP
        mask4.astype('uint8')
        mask2.astype('uint8')                           #Class_Destibution
        mask2=np.multiply(mask2,254)

        RESULT_IMAGE=np.concatenate((np.concatenate((mask2,mask3),axis=0),np.concatenate((mask4,im2show),axis=0)),axis=1)
        if vis:
            cv2.imshow('RESULT', RESULT_IMAGE)
            cv2.imshow('video', im2show)
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break

        if(test):
            if(len(AnomalyBoundary)>2):
                if(AnomalyStart==None):
                    AnomalyStart=AnomalyBoundary[0]
                    AnomalyStop=AnomalyBoundary[1]
                elif(AnomalyStop==FrameCounter and (AnomalyIndex+2)<AnomalyBoundary.amount):
                    AnomalyIndex+=2
                    AnomalyStart=AnomalyBoundary[AnomalyIndex]
                    AnomalyStop = AnomalyBoundary[AnomalyIndex+1]
            else:
                AnomalyStart = AnomalyBoundary[0]
                AnomalyStop = AnomalyBoundary[1]
            if (FrameCounter==AnomalyStart):
                AnomalyFrame=1
            if (FrameCounter==AnomalyStop):
                AnomalyFrame=0

            itemindex=np.where(mask4>0)
            size = itemindex[1].shape
            if (int(size[0])>10):
                if(AnomalyFrame==1):
                    TrueAnomaly_counter+=1
                else:
                    FalseAnomaly_counter+=1
            FrameCounter += 1
            if(AnomalyFrame==1):
                TotalAnomalyFrames+=1

        size=None
        if vid is None:
            if size is None:
                size = RESULT_IMAGE.shape[1], RESULT_IMAGE.shape[0]
            vid = cv2.VideoWriter('bestModelEver.avi', fourcc, float(24), size, True)
        vid.write(np.uint8(RESULT_IMAGE))
    if(test):
        print('result video',FrameCounter,TrueAnomaly_counter,FalseAnomaly_counter,TotalAnomalyFrames)
        print('True Positive',(TrueAnomaly_counter)/TotalAnomalyFrames,(FalseAnomaly_counter)/(FrameCounter - TotalAnomalyFrames))
        return destribution_object,destribution_motion,vid,((TrueAnomaly_counter)/TotalAnomalyFrames),((FalseAnomaly_counter)/(FrameCounter - TotalAnomalyFrames))
    else:
        return destribution_object, destribution_motion, vid

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.

  input_dir = args.load_dir #+ "/" + args.net + "/" + args.dataset
  print (input_dir)
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  pascal_classes = np.asarray(['__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor'])

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')

  # pdb.set_trace()

  print("load checkpoint %s" % (load_name))

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)

  if args.cuda > 0:
    cfg.CUDA = True

  if args.cuda > 0:
    fasterRCNN.cuda()

  fasterRCNN.eval()

  start = time.time()
  max_per_image = 100
  thresh = 0.05
  vis = True


#  print('Loaded Photo: {} images.'.format(num_images))

  vid = None
  MainFolder='video'
  videolist = sorted(os.listdir(MainFolder))
  for video in videolist:

      cap = cv2.VideoCapture(MainFolder+'/'+video)

      ret, previous = cap.read()
      rows, cols = previous.shape[0], previous.shape[1]

      print('create distribution map')

      destribution_motion = np.zeros((rows, cols, 2, 5))
      for i in range(rows):
          for j in range(cols):
              for f in range(2):
                  destribution_motion[i][j][f][4] = 3
      destribution_object = np.zeros((rows, cols, 22))
      destribution_object[:, :, 21] = np.add(destribution_object[:, :, 21], 1)

      previous = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)

      i=True
      mask2 = np.zeros((rows, cols, 3))
      counter=0
      destribution_object,destribution_motion,vid=combined(previous,rows,cols,destribution_motion,destribution_object,vid,False,None)

cv2.destroyAllWindows()
cap.release()
vid.release()


