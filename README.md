These scripts are used for convert datasets (MS COCO, Caltech pedestrian dataset) to PASCAL VOC format for later training.

### Requirements
- **Python2.7** (not work normally under python3)
- [MS COCO toolbox](https://github.com/pdollar/coco)
- cytoolz
- lmxl
- scipy, numpy

### Usage
#### COCO
`anno_json_image_urls.py`: extract image url (coco source not filckr) from annotation json file. See `anno_json_image_urls.sh` <br>
`download_coco_images.py`: download coco image files from given urls (extracted from instance/keypoint annotation json file) . See `download.sh`<br>
`anno_coco2voc.py`: convert coco annotation json file to VOC xml files. See `anno_coco2voc.sh` <br>
#### Caltech
`vbb2voc.py`: extract images with person bbox in seq file and convert vbb annotation file to xml files. <br>
**PS:** For Caltech pedestrian dataset, there are 4 kind of persons: `person`, `person-fa`, `person?`, `people`.
In my case, I just need to use `person` type data. If you want to use other types, specify `person_types` with
corresponding type list (like `['person', 'people']`) in `parse_anno_file` function.
#### HDA
`anno_had2voc`: convert HDA annotation info to VOC format.

