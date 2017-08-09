import os
from lxml import etree, objectify

# camera image size
cam_size = {
    "camera17": (640, 480),
    "camera18": (640, 480),
    "camera19": (640, 480),
    "camera40": (650, 480),
    "camera50": (1280, 800),
    "camera53": (1280, 800),
    "camera54": (1280, 800),
    "camera56": (1280, 800),
    "camera57": (1280, 800),
    "camera58": (1280, 800),
    "camera59": (1280, 800),
    "camera60": (2560, 1600)
}

def anno_file2dict(filename):
    annos = dict()
    with open(filename, 'r') as f:
        for line in f:
            if line:
                anno = {}
                items = line.strip().split(",")
                pic_name = str(items[0]) + "_" + str(items[1]) + ".jpg"
                pic_coord = items[2:6]
                pic_coord = [float(x) for x in pic_coord]
                if pic_name in annos:
                    annos[pic_name]["bbox"].append(pic_coord)
                else:
                    anno["filename"] = pic_name
                    anno["bbox"] = [pic_coord]
                    annos[pic_name] = anno
    return annos


def instance2xml_base(anno, bbox_type='xyxy'):
    """bbox_type: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)"""
    assert bbox_type in ['xyxy', 'xywh']
    camera_id = "camera" + str(anno["filename"].split("_")[0])
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('VOC2014_instance/person'),
        E.filename(anno["filename"]),
        E.source(
            E.database('HDA V1.3'),
            E.annotation('HDA V1.3'),
            E.image('HDA V1.3'),
            E.url('None')
        ),
        E.size(
            E.width(cam_size[camera_id][0]),
            E.height(cam_size[camera_id][1]),
            E.depth(3)
        ),
        E.segmented(0),
    )
    for bbox in anno['bbox']:
        if bbox_type == 'xyxy':
            xmin, ymin, w, h = bbox
            xmax = xmin+w
            ymax = ymin+h
        else:
            xmin, ymin, xmax, ymax = bbox
        E = objectify.ElementMaker(annotate=False)
        anno_tree.append(
            E.object(
            E.name("person"),
            E.bndbox(
                E.xmin(xmin),
                E.ymin(ymin),
                E.xmax(xmax),
                E.ymax(ymax)
            ),
            E.difficult(0)
            )
        )
    return anno_tree


def parse_anno_file(inputdir, outputdir):
    # annotation sub-directories in hda annotation input directory
    assert os.path.exists(inputdir)
    sub_dirs = os.listdir(inputdir)
    for sub_dir in sub_dirs:
        print "Parsing annotations of camera: ", sub_dir
        anno_file = os.path.join(inputdir, sub_dir, "Detections/allD.txt")
        annos = anno_file2dict(anno_file)
        outdir = os.path.join(outputdir, "Annotations", sub_dir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        for filename, anno in annos.items():
            anno_tree = instance2xml_base(anno)
            outfile = os.path.join(outdir, os.path.splitext(filename)[0]+".xml")
            print "Generating annotation xml file of picture: ", filename
            etree.ElementTree(anno_tree).write(outfile, pretty_print=True)

def visualize_bbox(xml_file, img_file):
    import cv2
    tree = etree.parse(xml_file)
    # load image
    image = cv2.imread(img_file)
    # get bbox
    for bbox in tree.xpath('//bndbox'):
        coord = []
        for corner in bbox.getchildren():
            coord.append(int(float(corner.text)))
        # draw rectangle
        # coord = [int(x) for x in coord]
        image = cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 2)
    # visualize image
    cv2.imshow("test", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    inputdir = "/startdt_data/HDA_Dataset_V1.3/hda_detections/GtAnnotationsAll"
    outputdir = "/startdt_data/HDA_Dataset_V1.3/hda_image_sequences_matlab"
    parse_anno_file(inputdir, outputdir)
    xml_file = "/startdt_data/HDA_Dataset_V1.3/hda_image_sequences_matlab/Annotations/camera17/17_2608.xml"
    img_file = "/startdt_data/HDA_Dataset_V1.3/hda_image_sequences_matlab/camera17/17_2608.jpg"
    visualize_bbox(xml_file, img_file)