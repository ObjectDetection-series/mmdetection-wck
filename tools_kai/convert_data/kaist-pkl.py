import os.path as osp
import mmcv
import numpy as np
from tools_kai.convert_data.utils import parse_xml
from tools_kai.convert_data.utils import track_progress_kai
import getpass


"""
Author:WangCk
Date:2019.12.3
Description: This script is used to load the train_annotations and test_annotations of kaist_mlfpn dataset
             to specified pkl file.
"""


def main():
    username = getpass.getuser()
    # xml_dir = osp.join('/home/' + username + '/WangCK/Data/datasets/kaist_mlfpn-rgbt/annotations-xml/')
    # pkl_dir = osp.join('/home/' + username + '/WangCK/Data/datasets/kaist_mlfpn-rgbt/annotations-pkl/')
    # txt_dir = osp.join('/home/' + username + '/WangCK/Data/datasets/kaist_mlfpn-rgbt/imageSets/')
    # img_dir = osp.join('/home/' + username + '/WangCK/Data/datasets/kaist_mlfpn-rgbt/images/')
    xml_dir = osp.join('/media/' + username + '/3rd/WangCK/Data/datasets/kaist_mlfpn-rgbt/annotations-xml/')
    pkl_dir = osp.join('/media/' + username + '/3rd/WangCK/Data/datasets/kaist_mlfpn-rgbt/annotations-pkl/')
    txt_dir = osp.join('/media/' + username + '/3rd/WangCK/Data/datasets/kaist_mlfpn-rgbt/imageSets/')
    img_dir = osp.join('/media/' + username + '/3rd/WangCK/Data/datasets/kaist_mlfpn-rgbt/images/')
    mmcv.mkdir_or_exist(pkl_dir)

    # all images
    train_filelist = osp.join(txt_dir, 'train-all-02.txt')
    img_all_names = mmcv.list_from_file(train_filelist)
    xml_all_paths = [osp.join(xml_dir, '{}.xml'.format(img_name)) for img_name in img_all_names]
    img_all_paths = [osp.join(img_dir, '{}.jpg'.format(img_name).replace('/I', '/visible/I')) for img_name in
                     img_all_names]
    xml_all_paths = np.array(xml_all_paths)
    img_all_paths = np.array(img_all_paths)
    # total_imgs = len(img_all_names)
    # permutation = np.random.permutation(total_imgs)
    # num_train = int(total_imgs * 0.9)       # ratio used to train:0.9

    # train images
    # idx_train = permutation[0:num_train + 1]
    xml_train_paths = xml_all_paths
    img_train_paths = img_all_paths
    flags = ['train' for _ in img_train_paths]
    train_annotations = track_progress_kai(parse_xml,
                                           list(zip(xml_train_paths, img_train_paths, flags)))
    mmcv.dump(train_annotations, osp.join(pkl_dir, 'train-all.pkl'))

    # validation images
    # idx_test = permutation[num_train + 1:]
    # xml_val_paths = xml_all_paths[idx_test]
    # img_val_paths = xml_all_paths[idx_test]
    # val_annotations = mmcv.track_progress(parse_xml,
    #                                       list(zip(xml_val_paths, img_val_paths)))
    # mmcv.dump(val_annotations, osp.join(pkl_dir, 'val.pkl'))

    # all test images
    test_filelist = osp.join(txt_dir, 'test-all-20.txt')
    img_test_names = mmcv.list_from_file(test_filelist)
    xml_test_paths = [osp.join(xml_dir, '{}.xml'.format(img_name)) for img_name in img_test_names]
    img_test_paths = [osp.join(img_dir, '{}.jpg'.format(img_name).replace('/I', '/visible/I')) for img_name in
                      img_test_names]
    flags = ['test' for _ in img_test_paths]
    test_annotations = track_progress_kai(parse_xml,
                                          list(zip(xml_test_paths, img_test_paths, flags)))
    mmcv.dump(test_annotations, osp.join(pkl_dir, 'test-all.pkl'))

    # day test images
    test_filelist = osp.join(txt_dir, 'test-day-20.txt')
    img_test_names = mmcv.list_from_file(test_filelist)
    xml_test_paths = [osp.join(xml_dir, '{}.xml'.format(img_name)) for img_name in img_test_names]
    img_test_paths = [osp.join(img_dir, '{}.jpg'.format(img_name).replace('/I', '/visible/I')) for img_name in
                      img_test_names]
    flags = ['test' for _ in img_test_paths]
    test_annotations = track_progress_kai(parse_xml,
                                          list(zip(xml_test_paths, img_test_paths, flags)))
    mmcv.dump(test_annotations, osp.join(pkl_dir, 'test-ady.pkl'))

    # night test images
    test_filelist = osp.join(txt_dir, 'test-night-20.txt')
    img_test_names = mmcv.list_from_file(test_filelist)
    xml_test_paths = [osp.join(xml_dir, '{}.xml'.format(img_name)) for img_name in img_test_names]
    img_test_paths = [osp.join(img_dir, '{}.jpg'.format(img_name).replace('/I', 'visible/I')) for img_name in
                      img_test_names]
    flags = ['test' for _ in img_test_paths]
    test_annotations = track_progress_kai(parse_xml,
                                          list(zip(xml_test_paths, img_test_paths, flags)))
    mmcv.dump(test_annotations, osp.join('test-night.pkl'))


if __name__ == '__main__':
    main()




