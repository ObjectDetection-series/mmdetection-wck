import xml.etree.ElementTree as ET
import json

"""
Author: Yuan Yuan
Date:2018/12/16
Location:SCU
"""


# create image element
def create_image(img_id=0, width=0, height=0, file_name="image_name"):
    image = {}
    image['id'] = img_id
    image['width'] = width
    image['height'] = height
    image['file_name'] = file_name
    image['flickr_url'] = None
    image['coco_url'] = None
    image['date_captured'] = None
    return image


# create annotation element
def create_annotation(ann_id=0, img_id=0, cat_id=1, height=0.0, ignore=0, occlusion=0,
                      area=None, bbox=[], iscrowd=0, vis_ratio=2):
    annotation = {}
    annotation['id'] = ann_id
    annotation['image_id'] = img_id
    annotation['category_id'] = cat_id
    # annotation['segmentation'] = seg
    annotation['height'] = height
    annotation['ignore'] = ignore
    annotation['occlusion'] = occlusion
    annotation['area'] = area
    annotation['bbox'] = bbox
    annotation['iscrowd'] = iscrowd
    annotation['vis_ratio'] = vis_ratio
    annotation['area'] = 10.0  # just a flag,  have no special meaning
    return annotation


def create_categories(cat_id=1, name='person', super_cat=None):
    cat = {}
    cat['id'] = cat_id
    cat['name'] = name
    cat['supercategory'] = super_cat
    return cat


def create_license(lic_id=1, name='public', url=None):
    lic = {}
    lic['id'] = lic_id
    lic['name'] = name
    lic['url'] = url
    return lic


class CoCoData(object):
    def __init__(self, xml_list, img_list, json_path):
        self.xml_list = xml_list
        self.img_list = img_list
        self.json_path = json_path
        self.AllInfo = {}
        self.info = {}
        self.images = []
        self.annotations = []
        self.licenses = []
        self.categories = []

        self.AllInfo['info'] = self.info
        self.AllInfo['images'] = self.images
        self.AllInfo['annotations'] = self.annotations
        self.AllInfo['licenses'] = self.licenses
        self.AllInfo['categories'] = self.categories

        self.info['year'] = 2015
        self.info['version'] = 'v1.0'
        self.info['description'] = 'kaist dataset with coco format'
        self.info['url'] = 'https://github.com/SoonminHwang/rgbt-ped-detection'
        self.info['date_created'] = '2015 CVPR'

        self.categories.append(create_categories())
        self.licenses.append(create_license())

        self.annId = 1
        self.imgId = 1

    def parse_xml(self, args):
        xml_path, img_path = args
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # for creating image element
        size = root.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)
        filename = img_path
        self.images.append(create_image(img_id=self.imgId, height=img_h, width=img_w, file_name=filename))
        # for creating annotation element
        for obj in root.findall('object'):
            name = obj.find('name').text
            bnd_box = obj.find('bndbox')
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            # width and height in COCO format
            bbox[2] = bbox[2] - bbox[0] + 1  # width
            bbox[3] = bbox[3] - bbox[1] + 1  # height
            # height field
            height = bbox[3]
            # ignore field
            # ignore = int(obj.find('difficult').text)
            iscrowd = 0
            # occlusion field
            occlusion = int(obj.find('occlusion').text)
            assert occlusion <= 2 and occlusion >= 0
            # visibility ratio
            vis_ratio = float(obj.find('vis_ratio').text)  # for caltech dataset
            # vis_ratio = float(2-occlusion)   # for kaist dataset
            # 'Reasonable' subset
            if name == 'person':
                self.annotations.append(create_annotation(
                    ann_id=self.annId, img_id=self.imgId, ignore=0,
                    occlusion=occlusion, height=height, bbox=bbox, vis_ratio=vis_ratio, iscrowd=iscrowd))
                # update annotation id
                self.annId += 1
        # update image id
        self.imgId += 1

    def convert(self):
        # traverse all xml files
        total_num = len(self.xml_list)
        for i, (xml_path, img_path) in enumerate(zip(self.xml_list, self.img_list)):
            self.parse_xml((xml_path, img_path))
            print("{}/{}".format(i, total_num))
        # write to disk
        json.dump(self.AllInfo, open(self.json_path, 'w'))
