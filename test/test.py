import numpy as np
import pandas as pd
from skimage import io
import cv2


from tqdm import tqdm
from mmcls.apis import init_model,inference_model


def mutil_threshold_Evaluate(results,thresholds,with_path=True,num_thd=1):
    """
    评估二分类模型结果，类别标签{0:异常,1:正常}
    args:
        results[List[list]]: 模型预测结果,N行包含N条结果，每一行包含[img_path,target,*scores]，其中score仅指异常类置信度。 
        thresholds[List]:  设置的异常样本阈值，异常置信度大于等于该阈值为异常类别。
        with_path[bool]:   results中是否包含图片路径，默认为True
        num_thd[int]:       需要存在多少个异常标签（区域），才认为是异常类。
    
    return:
        evaluation[DataFrame]: 在不同阈值下，模型的各项精度。
    """

    evaluation = {"阈值":[],"异常召回数":[],"异常误召数":[],"异常召回率":[],"异常精确率":[],"异常总数":[],
                  "正常召回数":[],"正常误召数":[],"正常召回率":[],"正常精确率":[],"正常总数":[]}
    for thd in thresholds:
        tn,fn,tp,fp = 0,0,0,0  # 翻拍：0  非翻拍：1
        for target,*scores in results:
            scores = list(map(float,scores))
            target = int(target)
            if sum([int(score>thd) for score in scores]) >= num_thd:
                label = 0
            else:
                label = 1

            if target == 0:
                if label == 0:
                    tp += 1
                else:
                    fn += 1
            if target == 1:
                if label == 1:
                    tn += 1
                else:
                    fp += 1
        print(f"tp,fn,tn,fp {tp,fn,tn,fp}")
        evaluation["阈值"].append(thd)
        evaluation["异常召回数"].append(tp)
        evaluation["异常召回率"].append(round(tp/(tp+fn)*100,2) if tp+fn > 0 else -1)
        evaluation["异常精确率"].append(round(tp/(tp+fp)*100,2) if tp+fp > 0 else -1)
        evaluation["异常总数"].append(tp+fn)
        evaluation["异常误召数"].append(fn)
        evaluation["正常召回数"].append(tn)
        evaluation["正常召回率"].append(round(tn/(tn+fp)*100,2) if tn+fp > 0 else -1)
        evaluation["正常精确率"].append(round(tn/(tn+fn)*100,2) if tn+fn > 0 else -1)
        evaluation["正常总数"].append(tn+fp)
        evaluation["正常误召数"].append(fp)
    return pd.DataFrame(evaluation)

class Classifier(object):
    def __init__(self,config,checkpoint) -> None:
        self.model = init_model(config,checkpoint)

    def inference(self,img):
        """
        args:
        x(str|narray):图片路径或者url或者一个通过BGR通道格式的numpy图片。
        
        return:
        score(float): 图片是异常类别[0]的置信度。
        """
        try:
            if isinstance(img,str):
                if img.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')): #是网路地址
                    im0 = io.imread(img)
                    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
                else:  #图片路径
                    im0 = cv2.imread(img)
            else:
                im0 = img

            pred = inference_model(self.model,img)
            light_score = pred["pred_score"] if pred["pred_label"] == 0 else 1-pred["pred_score"]
        except Exception as e:
            print(f"error occurs in model inference, info:{e}")
            return None
        else:
            return light_score
    
    def predict(self,anno):
        """
        args:
            anno[List[List]]: 分类模型标注，每一行包含[img_path,target]
        return:
            results[List[list]]: 模型预测结果,N行包含N条结果，每一行包含[img_path,target,*scores]，其中score仅指异常类置信度。 
        """
        result = []
        for path,target in anno:
            score = self.inference(path)
            if score:
                result.append({"path":path,"target":target,"score":score})
        
        return pd.DataFrame(result)








