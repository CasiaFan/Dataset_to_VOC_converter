#!/usr/bin/env bash
#python download_coco_images.py --input_file /startdt_data/COCO/dataset/image_urls/person_keypoints/train/person_keypoints_imag_urls.txt \
#                                   --output_dir /startdt_data/COCO/dataset/keypoints_train \
#                                   --n_workers 15
python download_coco_images.py --input_file /startdt_data/COCO/dataset/image_urls/person_keypoints/val/person_keypoints_imag_urls.txt \
                                   --output_dir /startdt_data/COCO/dataset/keypoints_val \
                                   --n_workers 15