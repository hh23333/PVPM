#!/bin/bash -x
cd /media/hh/disc_d/hh/code/openpose-master
path=/media/hh/disc_d/datasets/ICME2018_Occluded-Person-Reidentification_datasets/P_ETHZ/
image_dir=whole_body_images/
output_dir=whole_body_pose
# image_dir=occluded_body_images/
# output_dir=occluded_body_pose
files=$(ls ${path}${image_dir})
chmod +xw ./build/examples/openpose/openpose.bin
for dir in ${files}
do
    # mogrify -path ${path}${image_dir}${dir} -format jpg ${path}${image_dir}${dir}/*.tif
    ./build/examples/openpose/openpose.bin \
    --image_dir ${path}${image_dir}${dir} \
    --write_images ${path}${output_dir} \
    --model_pose COCO \
    --write_json ${path}${output_dir} \
    --heatmaps_add_parts true \
    --heatmaps_add_PAFs true \
    --write_heatmaps ${path}${output_dir} \
    --net_resolution -1x384
    --display 0
done