#!/usr/bin/env bash
python anno_coco2voc.py --anno_file /startdt_data/COCO/dataset/annotations/instances_train2014.json \
                        --type instance \
                        --output_dir /startdt_data/COCO/dataset/instance_train_annotation
python anno_coco2voc.py --anno_file /startdt_data/COCO/dataset/annotations/instances_val2014.json \
                        --type instance \
                        --output_dir /startdt_data/COCO/dataset/instance_val_annotation
#python anno_coco2voc.py --anno_file /startdt_data/COCO/dataset/annotations/person_keypoints_train2014.json \
#                        --type keypoint \
#                        --output_dir /startdt_data/COCO/dataset/keypoints_train_annotation
#python anno_coco2voc.py --anno_file /startdt_data/COCO/dataset/annotations/person_keypoints_val2014.json \
#                        --type keypoint \
#                        --output_dir /startdt_data/COCO/dataset/keypoints_val_annotation