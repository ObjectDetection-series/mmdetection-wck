import os.path as osp
import mmcv
from tools_kai.convert_data.utils import parse_xml
from tools_kai.convert_data.utils import track_progress_kai
import getpass

"""
Author:WangCK
Date:2019.12.2
Description: This script is used to load the train_annotation and test_annotation of Caltech dataset
             to specified pkl file.
"""


def main():
    username = getpass.getuser()
    xml_dir = osp.join('/home/' + username + '/WangCK/Data/datasets/Caltech/annotations-xml/')
    pkl_dir = osp.join('/home/' + username + '/WangCK/Data/datasets/Caltech/annotations-pkl/')
    txt_dir = osp.join('/home/' + username + '/WangCK/Data/datasets/Caltech/imageSets/')
    img_dir = osp.join('/home/' + username + '/WangCK/Data/datasets/Caltech/images')
    mmcv.mkdir_or_exist(pkl_dir)

    # all train images
    train_filelist = osp.join(txt_dir, 'train-all.txt')     # obtain the 'train-all.txt' file
    img_all_names = mmcv.list_from_file(train_filelist)     # obtain all names of train images
    xml_all_paths = [osp.join(xml_dir, img_name.replace('.txt', '.xml')) for img_name in img_all_names]
    img_all_paths = [osp.join(img_dir, img_name.replace('.txt', '.jpg')) for img_name in img_all_names]
    flags = ['train' for _ in img_all_names]
    train_annotations = track_progress_kai(parse_xml,
                                           list(zip(xml_all_paths, img_all_paths, flags)))
    mmcv.dump(train_annotations, osp.join(pkl_dir, 'train-all.pkl'))    # mmcv.dump类似于python的dump()，将结果保存到指定的.pkl文件

    # all test images
    test_filelist = osp.join(txt_dir, 'test-all.txt')       # obtain the 'test-all.txt' file
    img_all_names = mmcv.list_from_file(test_filelist)      # obtain all names of test images
    xml_all_paths = [osp.join(xml_dir, img_name.replace('.txt', '.xml')) for img_name in img_all_names]
    img_all_paths = [osp.join(img_dir, img_name.replace('.txt', '.jpg')) for img_name in img_all_names]
    flags = ['text' for _ in img_all_names]
    test_annotations = track_progress_kai(parse_xml,
                                          list(zip(xml_all_paths, img_all_paths, flags)))
    mmcv.dump(test_annotations, osp.join(pkl_dir, 'test-all.pkl'))      # mmcv.dump类似于pyhton的dump()，将结果保存到指定的.pkl文件


if __name__ == '__main__':
    main()








