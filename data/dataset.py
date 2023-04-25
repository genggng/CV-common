# 数据集划分
from skimage import io
import os
import pandas as pd
from tqdm import tqdm



def split_data(all_list,ratio=0.85):
    """
    args:
        all_list[str]:标签文件路径，每一行包含图片路径和对应标签，中间用空格分割。
    """

    all_list = pd.read_csv("./all_list.txt",sep=" ",header=None,names=['path','label'])
    train_list = all_list.sample(frac=0.85,random_state=5,axis=0)
    test_list = all_list[~all_list.index.isin(train_list.index)]

    print(f"共有数据{len(all_list)} 正样本(翻拍) {all_list.label.value_counts()[0]}  负样本（非翻拍）{all_list.label.value_counts()[1]} ")
    print(f"训练集划分比例为{ratio}")
    print(f"训练集数量 {len(train_list)}  正样本(翻拍) {train_list.label.value_counts()[0]}  负样本（非翻拍）{train_list.label.value_counts()[1]}")
    print(f"测试集数量 {len(test_list)}  正样本(翻拍) {test_list.label.value_counts()[0]}  负样本（非翻拍）{test_list.label.value_counts()[1]}")
    train_list.to_csv("train_list.txt",sep=" ",index=False,header=None)
    test_list.to_csv("test_list.txt",sep=" ",index=False,header=None)

def downlaod_img(urls,base_path):
    """
    args:
        urls[List[List]]: 数据标注，每一行包含图片URL和对应的标签。
        base_path: 图片要保存的路径。
    """
    all_image_list = []

    for i,(url,target) in enumerate(tqdm(urls)):
        try:
            image = io.imread(url)
        except Exception as e:
            print(e)
        else:
            sub_dir = "pos/" if target == 0 else "neg/"
            path = f"{os.path.join(base_path,sub_dir)}{i+1}.jpg"
            if image.shape[2] > 3:
                image = image[:,:,:3]
            io.imsave(path, image)
            all_image_list.append({"path":path,"target":target})

    all_image_list = pd.DataFrame(all_image_list)
    all_image_list.to_csv("./all_list.txt",sep=" ",index=False,header=False)


