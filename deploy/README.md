# 高光目标检测模型(stage1)
## 如何安装
1. 安装基础组件
```shell
pip install ultralytics
```
2. 下载yolov5仓库代码，并安装依赖
``` shell
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt  # install
```

## 如何使用
将本代码放入yolov5根目录下。
如果没有在项目根目录，需要在**detector.py**文件开头指定python路径。
```python
import sys
sys.path.append("/home/geng/yangugang/高光检测/yolov5")  # 添加yolov5所在路径
```

指定图片文件进行推理
```python
from detector import Detector

weight_path = "./best_card.pt"
img_path1 = "./高光.jpg"

model = Detector(weight=weight_path)
dets = model.inference(img_path1)
print(dets)
```
其中输入可以使图片路径或者opencv读取的numpy矩阵（读取通道格式BGR）

输出为一个[N,6]的矩阵，每一行代表一个预测框[x1,y1,x2,y2,score,label]