#!/usr/bin/env bash
python anno_json_image_urls.py --anno_file /startdt_data/COCO/dataset/annotations/instances_train2014.json \
                               --type instance \
                               --output_dir /startdt_data/COCO/dataset/image_urls/instance/train
python anno_json_image_urls.py --anno_file /startdt_data/COCO/dataset/annotations/instances_val2014.json \
                               --type instance \
                               --output_dir /startdt_data/COCO/dataset/image_urls/instance/val
python anno_json_image_urls.py --anno_file /startdt_data/COCO/dataset/annotations/person_keypoints_train2014.json \
                               --type keypoint \
                               --output_dir /startdt_data/COCO/dataset/image_urls/person_keypoints/train
python anno_json_image_urls.py --anno_file /startdt_data/COCO/dataset/annotations/person_keypoints_val2014.json \
                               --type keypoint \
                               --output_dir /startdt_data/COCO/dataset/image_urls/person_keypoints/val