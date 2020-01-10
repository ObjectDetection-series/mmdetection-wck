import numpy as np
from dataset import get_image_path
import scipy.misc as misc
import os

if __name__ == "__main__":
    file_list = ['/SSD64/LapSRN/DIV2K_train_pre/LR',
                 '/SSD64/LapSRN/DIV2K_train_pre/x2',
                 '/SSD64/LapSRN/DIV2K_train_pre/x4']

    for file_name in file_list:
        new_file_name = file_name+'_npy'
        img_paths = get_image_path(file_name)
        if not os.path.exists(new_file_name):
            os.makedirs(new_file_name)
            path_bar = tqdm(img_paths)
            for v in path_bar:
                img = misc.imread(v)
                ext = os.path.splitext(os.path.basename(v))[-1]
                name_sep = os.path.basename(v.replace(ext, '.npy'))
                np.save(os.path.join(new_file_name, name_sep), img)
        else:
            print("Binary file already exists, please confirm it in [%s]")
