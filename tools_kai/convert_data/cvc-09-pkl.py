import  os.path as osp
import mmcv
from tools_kai.convert_data.utils import parse_xml
from tools_kai.convert_data.utils import track_progress_kai
import getpass

"""
Author:WangCK
Date:2019.11.28
Description: This script is used to load the train_annotations and test_annotations of cvc09 dataset
             to specified pkl file.
"""


def main():
    username = getpass.getuser()
    xml_dir = osp.join('/home/' + username + '/WangCK/Data/datasets/CVC/annotations-xml/')
    pkl_dir = osp.join('/home/' + username + '/WangCK/Data/datasets/CVC/annotations-pkl/')
    txt_dir = osp.join('/home/' + username + '/WangCK/Data/datasets/CVC/imageSets/')
    img_dir = osp.join('/home/' + username + '/WangCK/Data/datasets/CVC/images')
    mmcv.mkdir_or_exist(pkl_dir)

    # all train images
    train_filelist = osp.join(txt_dir, 'train-all.txt')
    img_all_names = mmcv.list_from_file(train_filelist)
    xml_paths = [osp.join(xml_dir, img_name.replace('.txt', '.xml')) for img_name in img_all_names]
    img_paths = [osp.join(img_dir, img_name.replace('.txt', '.jpg')) for img_name in img_all_names]
    flags = ['train' for _ in img_all_names]
    train_annotations = track_progress_kai(parse_xml, list(zip(xml_paths, img_paths, flags)))
    mmcv.dump(train_annotations, osp.join(pkl_dir, 'train-all.pkl'))

    # all test images
    test_filelist = osp.join(txt_dir, 'test-all.txt')
    img_all_names = mmcv.list_from_file((test_filelist))
    xml_paths = [osp.join(xml_dir, img_name.replace('.txt', '.xml')) for img_name in img_all_names]
    img_paths = [osp.join(img_dir, img_name.replace('.txt', '.jpg')) for img_name in img_all_names]
    flags = ['test' for _ in img_all_names]
    test_annotations = track_progress_kai(parse_xml, list(zip(xml_paths, img_paths, flags)))
    mmcv.dump(test_annotations, osp.join(pkl_dir, 'test-all.pkl'))


if __name__ == '__main__':
    main()
