from match import pickle_load
from table_recognition.utils import xywh2xyxy
import os
import cv2
import numpy as np


base_path = '/home/zhaohj/Documents/dataset/Table/TAL/val'
structure_master_file = '/home/zhaohj/Documents/workspace/github_my/TableMASTER-mmocr/structure_master_result_folder/structure_master_results_0.pkl'
structure_master_results = pickle_load(structure_master_file, prefix='structure')
for res_name in structure_master_results:
    res = structure_master_results[res_name]
    bboxes = res['bbox']
    bboxes = xywh2xyxy(bboxes)
    img = cv2.imread(os.path.join(base_path, res_name))
    for bbox in bboxes:
        if bbox.sum() == 0.:
            continue
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])), (0, 255, 0), thickness=1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
# structure_master_result = structure_master_results[file_name]