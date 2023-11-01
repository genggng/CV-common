# %%
import fiftyone as fo
import fiftyone.zoo as foz

# %% [markdown]
# ### 1. 加载visdrone val数据集

# %%
visdrone_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path="/home/wsg/dataset/VisDrone_Dataset_COCO_Format/images/instances_UAVval",
    labels_path="/home/wsg/dataset/VisDrone_Dataset_COCO_Format/annotations/instances_UAVval.json",
    label_types=["detections"],
    include_id=True,
)

# %%
print(visdrone_dataset.default_classes)
print(visdrone_dataset)

# %% [markdown]
# ### 2. 添加模型预测结果

# %%
import mmcv

# 加载模型输出
reppoint_result =  mmcv.load("/home/wsg/sod/coco_tools/results/reppoints_v2_visdrone_val_results-nms320.bbox.json")
ufpmp_result = mmcv.load("/home/wsg/sod/coco_tools/results/ufpmp_visdrone_val_results.bbox.json")


for i in range(len(reppoint_result)):
    reppoint_result[i]["image_id"] += 1
for i in range(len(ufpmp_result)):
    ufpmp_result[i]["image_id"] += 1

# %%
import fiftyone.utils.coco as fouc
# 添加输出到dataset
classes = visdrone_dataset.default_classes
fouc.add_coco_labels(visdrone_dataset, "reppoint_pred", reppoint_result, classes)
fouc.add_coco_labels(visdrone_dataset, "ufpmp_pred", ufpmp_result, classes)

# %%
session = fo.launch_app(visdrone_dataset,port=5151)
session.wait()

