import json
import cytoolz
import argparse, os, re


def extract_urls(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    content = json.load(open(args.anno_file, 'r'))
    merge_info_list = map(cytoolz.merge, cytoolz.join('id', content['images'], 'image_id', content['annotations']))
    if args.type == 'instance':
        outfiles = {category['id']: os.path.join(args.output_dir, re.sub(" ", "_", category['name'])+"_image_urls.txt") for category in content['categories']}
        for info in merge_info_list:
            print "Saving file name: ", info['file_name']
            with open(outfiles[info['category_id']], "a") as f:
                f.write(os.path.splitext(info['file_name'])[0]+" "+info['coco_url']+"\n")
            f.close()
        print "Exporting coco image urls for instance done!"
    else:
        outfile = os.path.join(args.output_dir, "person_keypoints_imag_urls.txt")
        url_dict = {info['file_name']: info['coco_url'] for info in merge_info_list}
        with open(outfile, "w") as f:
            for name, url in url_dict.items():
                f.write(os.path.splitext(name)[0]+" "+url+"\n")
        f.close()
        print "Exporting coco image urls for keypoints done!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_file", help="annotation file for object instance/keypoint")
    parser.add_argument("--type", type=str, help="object instance or keypoint", choices=['instance', 'keypoint'])
    parser.add_argument("--output_dir", help="output directory for image urls in json annotation file")
    args = parser.parse_args()
    extract_urls(args)