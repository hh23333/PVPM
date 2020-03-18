#!/bin/bash -x
cd PATH_TO_openpose-master/
path=PATH_TO_YOUR_DATASET_ROOT/Market-1501-v15.09.15/

image_dir=bounding_box_train/
output_dir=bounding_box_pose_train

./build/examples/openpose/openpose.bin \
--image_dir ${path}${image_dir} \
--write_images ${path}${output_dir} \
--model_pose COCO \
--write_json ${path}${output_dir} \
--heatmaps_add_parts true \
--heatmaps_add_PAFs true \
--write_heatmaps ${path}${output_dir} \
--net_resolution -1x384

image_dir=bounding_box_test/
output_dir=bounding_box_pose_test

./build/examples/openpose/openpose.bin \
--image_dir ${path}${image_dir} \
--write_images ${path}${output_dir} \
--model_pose COCO \
--write_json ${path}${output_dir} \
--heatmaps_add_parts true \
--heatmaps_add_PAFs true \
--write_heatmaps ${path}${output_dir} \
--net_resolution -1x384

image_dir=query/
output_dir=query_pose

./build/examples/openpose/openpose.bin \
--image_dir ${path}${image_dir} \
--write_images ${path}${output_dir} \
--model_pose COCO \
--write_json ${path}${output_dir} \
--heatmaps_add_parts true \
--heatmaps_add_PAFs true \
--write_heatmaps ${path}${output_dir} \
--net_resolution -1x384
