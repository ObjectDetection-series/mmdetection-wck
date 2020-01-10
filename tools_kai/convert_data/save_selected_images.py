import os.path as osp
import getpass
import mmcv
import cv2
import io
import matplotlib.pyplot as plt


"""
Author:WangCk
Date:2019.1.9
Description: This py file is used to select the images that have pedestrian. (7380 / 25086)
"""


def main():
    username = getpass.getuser()
    txt_dir = osp.join('/media/' + username + '/3rd/WangCK/Data/datasets/kaist-rgbt/imageSets/')
    img_dir = osp.join('/media/' + username + '/3rd/WangCK/Data/datasets/kaist-rgbt/images/')

    # true training thremal images
    train_filelist = osp.join(txt_dir, 'true-test-all-20.txt')
    with open(train_filelist) as f:
        lines = f.readlines()

    prefix = '/media/' + username + '/3rd/WangCK/Data/datasets/kaist-rgbt/saliencyMaps/test_lwir/'
    for line in lines:
        # load thermal image
        img_t_path = line.rstrip()    # 去除后边的换行符\n
        # print(img_t_path)
        img_t = cv2.imread(img_t_path)

        # save thermal to specified path
        pre_savapath = osp.join(prefix, img_t_path[-26:-10])
        mmcv.mkdir_or_exist(pre_savapath)

        save_name = osp.join(pre_savapath, img_t_path[-10:])
        print(save_name)
        cv2.imwrite(save_name, img_t)


if __name__ == '__main__':
    main()


















