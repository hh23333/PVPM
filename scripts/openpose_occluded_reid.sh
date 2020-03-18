#!/bin/bash -x
cd PATH_TO_openpose-master/
path=PATH_TO_YOUR_DATASET_ROOT/ICME2018_Occluded-Person-Reidentification_datasets/Occluded_REID/
image_dir=whole_body_images/
output_dir=whole_body_pose
# image_dir = occluded_body_images
files=$(ls ${path}${image_dir})
chmod +xw ./build/examples/openpose/openpose.bin
for dir in ${files}
do
    mogrify -path ${path}${image_dir}${dir} -format jpg ${path}${image_dir}${dir}/*.tif
    ./build/examples/openpose/openpose.bin \
    --image_dir ${path}${image_dir}${dir} \
    --write_images ${path}${output_dir} \
    --model_pose COCO \
    --write_json ${path}${output_dir} \
    --heatmaps_add_parts true \
    --heatmaps_add_PAFs true \
    --write_heatmaps ${path}${output_dir} \
    --net_resolution -1x384
done

image_dir=occluded_body_images/
output_dir=occluded_body_pose
files=$(ls ${path}${image_dir})
chmod +xw ./build/examples/openpose/openpose.bin
for dir in ${files}
do
    mogrify -path ${path}${image_dir}${dir} -format jpg ${path}${image_dir}${dir}/*.tif
    ./build/examples/openpose/openpose.bin \
    --image_dir ${path}${image_dir}${dir} \
    --write_images ${path}${output_dir} \
    --model_pose COCO \
    --write_json ${path}${output_dir} \
    --heatmaps_add_parts true \
    --heatmaps_add_PAFs true \
    --write_heatmaps ${path}${output_dir} \
    --net_resolution -1x384
done




