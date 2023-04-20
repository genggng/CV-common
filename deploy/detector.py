import torch
import sys
sys.path.append("/home/geng/yangugang/高光检测/yolov5")  # 添加yolov5所在路径

from models.common import DetectMultiBackend
from utils.general import (check_img_size, cv2,non_max_suppression, scale_boxes, xyxy2xywh,xywh2xyxy,clip_boxes)
from utils.torch_utils import select_device
from utils.augmentations import letterbox
import numpy as np


class Detector(object):
    def __init__(self,
        weight,  # yolov5模型权重路径，必须
        data=None,               
        imgsz=(640, 640), # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        gain=1.02,       #将检测框放大gain倍
        pad=10,          # 检测框pading
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
        self.gain = gain
        self.pad = pad

        self.model = DetectMultiBackend(weight, device=self.device, dnn=dnn, data=data, fp16=half)
        self.stride, self.pt = self.model.stride, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)
        self.bs = 1  #目前只支持单bs推理和处理
        # model warmup
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else self.bs, 3, *self.imgsz))  # warmup



    def inference(self,img):
        """
        args:
            img(str|ndarray) : 图片路径或者opencv读取的图片numpy数组。
        
        return:
            dets(tensor[N*6]):返回对图片的检测结果，共有N个检测框，组织形式为[x1,y1,x2,y2,score,label]
        """
        if isinstance(img,str):
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
        
        dets = pred[0]
        dets[:, :4] = scale_boxes(im.shape[2:], dets[:, :4], im0.shape).round()

        # 将检测框稍微放缩下
        for i,(*xyxy,score,label) in enumerate(dets):
            xyxy = torch.tensor(xyxy).view(-1, 4)
            b = xyxy2xywh(xyxy)  # boxes
            b[:, 2:] = b[:, 2:] * self.gain + self.pad  # box wh * gain + pad
            xyxy = xywh2xyxy(b).long()
            clip_boxes(xyxy, im0.shape)
            dets[i,:4] = xyxy
        return dets

if __name__ == "__main__":
    weight_path = "./best_card.pt"
    img_path1 = "./高光.jpg"

    model = Detector(weight=weight_path)
    dets = model.inference(img_path1)
    
    print(dets)

    