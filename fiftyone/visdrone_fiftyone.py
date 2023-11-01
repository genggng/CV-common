import fiftyone as fo
import fiftyone.zoo as foz

visdrone_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path="/home/wsg/dataset/visdrone-2019/val/images",
    labels_path="/home/wsg/dataset/visdrone-2019/val/annotations/glsan_val.json",
    include_id=True,
    max_sample=25,
)
session = fo.launch_app(visdrone_dataset)

# visdrone_ufp_dataset = fo.Dataset.from_dir(
#     dataset_type=fo.types.COCODetectionDataset,
#     data_path="/home/wsg/dataset/VisDrone_Dataset_COCO_Format/images/instances_UAVval",
#     labels_path="/home/wsg/dataset/VisDrone_Dataset_COCO_Format/annotations/instances_UAVval.json",
#     include_id=True,
# )
# session2 = fo.launch_app(visdrone_ufp_dataset)