from unittest import result
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def eval_map(gt_ann_path,dt_result_path,maxDets=None):
    coco_gt= COCO(gt_ann_path)
    coco_dt = coco_gt.loadRes(dt_result_path)
    imgIds = coco_gt.getImgIds()

    cocoEval = COCOeval(coco_gt,coco_dt,"bbox")
    cocoEval.params.imgIds  = imgIds
    if maxDets:
        cocoEval.params.maxDets=maxDets
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == "__main__":
    annFile='/home/wsg/dataset/visdrone-2019/val/annotations/glsan_val.json'
    # resultFile='/home/wsg/sod/coco_tools/results/reppoints_v2_visdrone_val_results-nms320.bbox.json'
    resultFile='/home/wsg/sod/glsan/train_log/1108_cascade_rcnn_res50_visdrone/inference/coco_instances_results.json'
    # maxDets = [1,1,100]
    maxDets = None
    eval_map(annFile,resultFile,maxDets)