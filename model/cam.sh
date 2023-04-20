img_name="neg_nei.png"
category="1"
config="efficient-b7_live.py"
checkpoint="efficient-b7_e50_thiy.pth"
python ../tools/visualizations/vis_cam.py \
    ./images/$img_name \
    $config \
    $checkpoint \
    --method GradCAM \
    --save-path ./images/${category}-cam-$img_name \
    --target-category $category 
    # GradCAM++, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM