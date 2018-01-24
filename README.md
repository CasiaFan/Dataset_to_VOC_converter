These scripts are used for convert datasets (MS COCO, Caltech pedestrian dataset) to PASCAL VOC format for later training.

### Requirements
- **Python2.7** (not work normally under python3)
- [MS COCO toolbox](https://github.com/pdollar/coco)
- cytoolz
- lmxl
- scipy, numpy

### Usage
`anno_json_image_urls.py`: extract image url (coco source not filckr) from annotation json file. See `anno_json_image_urls.sh` <br>
`download_coco_images.py`: download coco image files from given urls (extracted from instance/keypoint annotation json file) . See `download.sh`<br>
`anno_coco2voc.py`: convert coco annotation json file to VOC xml files. See `anno_coco2voc.sh` <br>
`vbb2voc.py`: extract images with person bbox in seq file and convert vbb annotation file to xml files. <br>

