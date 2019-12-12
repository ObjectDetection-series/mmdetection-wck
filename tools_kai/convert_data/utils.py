import xml.etree.ElementTree as ET
import numpy as np
import collections
import sys
from mmcv.utils.progressbar import ProgressBar


def parse_xml(args):
    xml_path, img_path, flag = args
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        difficult = int(obj.find('difficult').text)  # 'difficult' always be zero because of original code
        occlusion = int(obj.find('occlusion').text)

        bnd_box = obj.find('bndbox')
        bbox = [
            int(float(bnd_box.find('xmin').text)),
            int(float(bnd_box.find('ymin').text)),
            int(float(bnd_box.find('xmax').text)),
            int(float(bnd_box.find('ymax').text))
        ]
        if name != 'person' or difficult > 0 or occlusion > 0 or (bbox[3] - bbox[1] + 1) < 50:
            bboxes_ignore.append(bbox)
            labels_ignore.append(0)
        else:
            bboxes.append(bbox)
            labels.append(1)

    if not bboxes:
        if flag == 'train':
            return None  # images without pedestrian can be ignored during training
        else:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
    else:
        bboxes = np.array(bboxes, ndmin=2)
        labels = np.array(labels)

    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0,))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
        labels_ignore = np.array(labels_ignore)

    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'flag': 0,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation


"""
Author:Yuan Yuan
Date:2019/03/08
Description:Prepare datasets for Faster RCNN which deals with cross-model 
"""


def parse_xml_cross(args):
    xml_path, img_path, flag, flag_model = args
    annotation = parse_xml((xml_path, img_path, flag))
    if annotation is None:
        return None
    annotation['flag'] = flag_model
    return annotation


"""
Author:Yuan Yuan
Date:2019/03/05
Description:Prepare datasets for auto-encoder
"""


def parse_xml_coder(args):
    xml_path, img_path, flag_coder = args
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'flag': flag_coder,
        'ann': {
            'bboxes': None,
            'labels': None,
            'bboxes_ignore': None,
            'labels_ignore': None
        }
    }
    return annotation


def track_progress_kai(func, tasks, bar_width=50, **kwargs):
    """Track the progress of tasks execution with a progress bar(进度条).

    Tasks are done with a simple for-loop.

    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        bar_width (int): Width of progress bar.

    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], collections.Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, collections.Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    prog_bar = ProgressBar(task_num, bar_width)
    results = []
    for task in tasks:
        temp = func(task, **kwargs)
        if temp is not None:
            results.append(temp)
            prog_bar.update()
    sys.stdout.write('\n')
    return results
