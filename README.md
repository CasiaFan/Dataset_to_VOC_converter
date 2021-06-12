These scripts are used for converting annotations of pedestrian datasets, including MS COCO, Caltech pedestrian dataset, and HDA to PASCAL VOC format.

### Requirements
- **Python3.X**
- [MS COCO toolbox](https://github.com/pdollar/coco)
- cytoolz
- pathos 
- lmxl
- scipy, numpy

### Usage
#### COCO
`anno_json_image_urls.py`: extract image url (coco source not filckr) from annotation json file. See `anno_json_image_urls.sh` <br>
`download_coco_images.py`: download coco image files from given urls (extracted from instance/keypoint annotation json file) . See `download.sh`<br>
`anno_coco2voc.py`: convert coco annotation json file to VOC xml files. See `anno_coco2voc.sh` <br>

For exmaple:
```bash
python3 anno_coco2voc.py --anno_file=/Path/to/instances_train2014.json \
                         --type=instance \
                         --output_dir=/Path/to/instance_anno_dir
```

#### Caltech
`vbb2voc.py`: extract images with person bbox in seq file and convert vbb annotation file to xml files. <br>
**PS:** For Caltech pedestrian dataset, there are 4 kind of persons: `person`, `person-fa`, `person?`, `people`.
In my case, I just need to use `person` type data. If you want to use other types, specify `person_types` with
corresponding type list (like `['person', 'people']`) in `parse_anno_file` function.

```bash
python3 vbb2voc.py --seq_dir=path/to/caltech/seq/dir \        
                   --vbb_dir=path/to/caltech/vbb/dir \
                   --output_dir=/output/saving/path \
                   --person_type=person
```

#### HDA
`anno_had2voc`: convert HDA annotation info to VOC format.
```bash
python3 anno_hda2voc.py --input_dir=path/to/HDA_Dataset_V1.3/hda_detections/GtAnnotationsAll \ 
                        --output_dir=anno/saving/path
```
