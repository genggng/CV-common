import pandas as pd
import cv2
import os
import numpy as np


def yolo_to_voc(size, box):
    # size = (w,h)
    x = box[0] * size[0]
    w = box[2] * size[0]
    y = box[1] * size[1]
    h = box[3] * size[1]
    xmin = int(x - w/2)
    xmax = int(x + w/2)
    ymin = int(y - h/2)
    ymax = int(y + h/2)
    return (xmin, ymin, xmax, ymax)

def make_crops_from_yolo(class_names,base_path,flag):
    """
    从yolo标注中提取每个instance的图片和类别
    args:
        class_names[List]:  类别名列表
        base_path[str]: 原始数据地址
        flag[str]:  trian or test 
    """


    img_dir = os.path.join(base_path,"images")
    label_dir = os.path.join(base_path,"labels")
    patch_dir = os.path.join(base_path,"patch")

    for i,cls in enumerate(class_names):
        cls_dir = os.path.join(patch_dir,cls)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir)

    patch_anno = []  # [path,label]  label in [0,17]

    for img_name in os.listdir(img_dir):
        name = os.path.splitext(img_name)[0]
        anno_name = name + ".txt"
        patch_name = name+".png"
        img_path = os.path.join(img_dir,img_name)
        img = cv2.imread(img_path) 
        h,w,_ = img.shape
        # 读取坐标
        with open(os.path.join(label_dir,anno_name),"r") as f:
            for line in f.readlines():
                label,*box = list(map(float,line.strip().split()))
                label = int(label)
                if label == 14 or label == 15:  #跳过14和15号区域
                    continue
                
                xyxy = yolo_to_voc((w,h),box)
                crop = img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
                # p_h,p_w,_ = crop.shape
                # if p_h > p_w:  #  旋转为w>h
                #     crop=np.rot90(crop)
                save_path = os.path.join(f"{patch_dir}/{class_names[label]}",patch_name)

                try:
                    cv2.imwrite(save_path,crop)
                except Exception:
                    print(save_path,crop)
                else:
                    patch_anno.append({"path":save_path,"label":label})



    patch_anno_df = pd.DataFrame(patch_anno, columns=['path', 'label'])
    patch_anno_df.to_csv(f"{base_path}/patch_{flag}.txt",sep = ' ', index=False,header=False)

class_names = ['learner_mark', 'greenhand_mark', 'other_mark']
flag="train"
base_path = f"/ai/wsg/data/learn_car/yolo-format/{flag}"

make_crops_from_yolo(class_names,base_path,flag)