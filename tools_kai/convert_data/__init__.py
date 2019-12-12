"""
Author:WangCK
Date:2019.12.6
Description: This python package is used to convert xml datasets format to pkl or coco format.
             In this project, pkl format is used.
"""
from .voc_to_coco import CoCoData
from .utils import parse_xml, parse_xml_coder, track_progress_kai

__all__ = ['CoCoData', 'parse_xml', 'parse_xml_coder', 'track_progress_kai']
