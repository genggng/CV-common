import onnxruntime
import cv2
import numpy as np

def preprocess(dets,im0,mean=[123.675, 116.28, 103.53],std=[58.395, 57.12, 57.375]):
    """
    args:
        dets([N,6]): 检测结果，为一个[N,6]的矩阵，代表N个检测框。每个检测框组织为[x1,y1,x2,y2,score,label]
        im0: 原始图片的路径，或者一个opencv读取的numpy数组，通道格式为BGR。
        gain: 检测框的放缩倍数 
        pad:  对检测框进行增大
        mean: 标准化时的均值
        std: 标准化时的标准差
    
    return:
        crops_temsor([N,3,120,120]): N个预处理后的图片patch
    """
    crops_tensor = []  #把图片中N个patch数据拼成一个batch

    if isinstance(im0,str):
        im0 = cv2.imread(im0)

    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB) #需要转为RGB格式

    for *xyxy,score,label in dets:
        xyxy = np.array(xyxy).reshape(-1,4)
        crop = im0[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), :]  #获取原图上裁切

        h,w,_ = crop.shape
        if h > w:  #  旋转为w>h
            crop=np.rot90(crop)
            h,w = w,h
        # print(f"原始尺寸{crop.shape[:2]}")
        
        width = 120   #固定宽度为120
        height = int(width/w * h)  #高度进行相同倍率的放缩
        crop = cv2.resize(crop,(width,height))   # 保持长宽比的resize
        # print(f"放缩尺寸{crop.shape[:2]}")
        
        crop = cv2.copyMakeBorder(crop,0,width-height,0,0,cv2.BORDER_CONSTANT,value=0)        
        # print(f"pading后尺寸 {crop.shape[:2]}")
        
        
        crop = (crop - mean) /std   # 标准化
        crop = crop.transpose(2, 0, 1).astype(np.float32)  #转化通道顺序为C*H*W

        crops_tensor.append(crop)
    return np.array(crops_tensor)




# 原始图片路径
img_path1 = "./高光.jpg"

# dets来自目标检测模型的预测结果，形式为[N，6]的二维数组。[x1,y1,x2,y2,score,label]
dets = np.array([
        [323.00000, 250.00000, 445.00000, 287.00000,   0.95583,   2.00000],
        [198.00000, 321.00000, 318.00000, 356.00000,   0.95095,   6.00000],
        [340.00000, 288.00000, 406.00000, 323.00000,   0.94992,   4.00000],
        [345.00000, 320.00000, 466.00000, 356.00000,   0.94587,   8.00000],
        [296.00000,  92.00000, 505.00000, 127.00000,   0.94175,   0.00000]
        ])


# 数据预处理
input_tensor = preprocess(dets,img_path1)


# 加载onnx模型
model_path  = "./shufflenetV2_thiy.onnx"
onnxruntime.set_default_logger_severity(3)
sess = onnxruntime.InferenceSession(path_or_bytes=model_path, providers=['CUDAExecutionProvider'])
input_placeholder_name = sess.get_inputs()[0].name
result = []
for x in input_tensor:
    x = np.expand_dims(x,axis=0)
    raw_result = sess.run(input_feed={input_placeholder_name: x},output_names=None)
    result.append(raw_result[0][0][0])

print(result)  # [N，1] 预测每个检测框区域是高光的置信度。[0.0262456, 1.7461214e-05, 1.4605854e-08, 0.8106861, 5.206746e-06]