#!/bin/bash -x
cd /media/hh/disc_d/hh/code/openpose-master
path=/media/hh/disc_d/datasets/ICME2018_Occluded-Person-Reidentification_datasets/Occluded_Duke/
for split in 'bounding_box_train' 'bounding_box_test' 'query'
do
    # image_dir = occluded_body_images
    output_dir=${split}'_pose'
    echo ${output_dir}
    chmod +xw ./build/examples/openpose/openpose.bin
    ./build/examples/openpose/openpose.bin \
    --image_dir ${path}${split} \
    --write_images ${path}${output_dir} \
    --model_pose COCO \
    --write_json ${path}${output_dir} \
    --heatmaps_add_parts true \
    --heatmaps_add_PAFs true \
    --write_heatmaps ${path}${output_dir} \
    --net_resolution -1x384
    --display 0
done




