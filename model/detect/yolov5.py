import torch
import sys
sys.path.append("/home/geng/yangugang/高光检测/yolov5")  # 添加yolov5所在路径

from models.common import DetectMultiBackend
from utils.general import (check_img_size, cv2,non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import numpy as np

from utils.general import clip_boxes
import numpy as np

import pandas as pd
from skimage import io
from tqdm import tqdm

import torch
from utils.general import cv2
from mmcls.apis import init_model,inference_model
import os

class Classifier(object):
    def __init__(self,config,checkpoint) -> None:
        self.model = init_model(config,checkpoint)

    def inference(self,img):
        """
        args:
        x(str|narray):图片路径或者一个通过BGR通道格式的numpy图片。
        
        return:
        score(float): 图片是高光类别的置信度。
        """
        if isinstance(img,str):
            img=cv2.imread(img)
        h,w,_  = img.shape
        if h > w:  #
            img=np.rot90(img)
        pred = inference_model(self.model,img)
        light_score = pred["pred_score"] if pred["pred_label"]%2 == 1  else 1-pred["pred_score"]

        return light_score


class Detector(object):
    def __init__(self,
        weight,  # yolov5模型权重路径，必须
        apply_cls = False,  #是否添加分类模型
        cls_config=None,
        cls_checkpoint=None,
        data=None,               
        imgsz=(640, 640), # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device = '',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,         
        ) -> None:
        
        self.data = data
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = select_device(device)
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.half = half
        self.dnn = dnn
        self.vid_stride = vid_stride
        self.apply_cls = apply_cls


        self.model = DetectMultiBackend(weight, device=self.device, dnn=dnn, data=data, fp16=half)
        self.stride, self.pt = self.model.stride, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)
        self.bs = 1  #目前只支持单bs推理和处理
        # model warmup
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else self.bs, 3, *self.imgsz))  # warmup

        if self.apply_cls:
            self.cls_model = Classifier(cls_config,cls_checkpoint)

    def apply_classifier(self, cls_model,det,im0):
        scores = []
        for *xyxy, conf, cls in det:
            xyxy = torch.tensor(xyxy).view(-1, 4)
            clip_boxes(xyxy, im0.shape)
            crop = im0[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), :]
            score = cls_model.inference(crop)
            scores.append(score)
        return scores

    def inference(self,img):
        """
        img(str|ndarray) : 图片路径或者opencv读取的图片numpy数组。
        """
        if isinstance(img,str):
            if img.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')): #是网路地址
                im0 = io.imread(img)
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
            else:  #图片路径
                im0 = cv2.imread(img)
        else:
            im0 = img
        
        # 数据预处理
        im = letterbox(im0, self.imgsz, stride=self.stride, auto=self.pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous  
        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        # 模型推理与NMS
        pred = self.model(im, augment=self.augment, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        
        det = pred[0]

        
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        # 屏蔽掉id=14和id=15的区域，即'v_use_nature_pass', 'v_use_nature_nopass'
        res = []
        for d in det:
            if int(d[5]) != 14 and  int(d[5]) != 15:
                res.append(d.cpu().numpy())
        res = np.array(res)
        if self.apply_cls:
            scores = self.apply_classifier(self.cls_model,res,im0)
            res[:,4] = np.array(scores)
            res[:,5] = np.zeros(scores.shape)
        return res

    

