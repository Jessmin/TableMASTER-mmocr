from match import pickle_load
from table_recognition.utils import xywh2xyxy
import os
import cv2
import numpy as np
from table_inference import Structure_Recognition
from table_recognition.utils import structure_visual

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    det_model_dir=
    '/home/zhaohj/Documents/checkpoint/paddOCR/inference/ch_ppocr_server_v2.0/det',
    rec_model_dir=
    '/home/zhaohj/Documents/checkpoint/paddOCR/inference/ch_ppocr_server_v2.0/rec',
    rec_char_dict_path=
    '/home/zhaohj/Documents/checkpoint/paddOCR/inference/ch_ppocr_server_v2.0/ppocr_keys_v1.txt',
    cls_model_dir=
    '/home/zhaohj/Documents/checkpoint/paddOCR/inference/ch_ppocr_server_v2.0/cls',
    use_angle_cls=True,
    max_text_length=15,
    drop_score=0.6,
    det_db_unclip_ratio=2.0,
    lang="ch")

base_path = '/home/zhaohj/Documents/dataset/Table/TAL/val'
structure_master_file = '/home/zhaohj/Documents/workspace/github_my/TableMASTER-mmocr/structure_master_result_folder/structure_master_results_0.pkl'
structure_master_results = pickle_load(
    structure_master_file, prefix='structure')
for res_name in structure_master_results:
    res = structure_master_results[res_name]
    bboxes = res['bbox']
    bboxes = xywh2xyxy(bboxes)
    img = cv2.imread(os.path.join(base_path, res_name))
    for bbox in bboxes:
        if bbox.sum() == 0.:
            continue
        img = cv2.rectangle(
            img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            thickness=1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
# structure_master_result = structure_master_results[file_name]


# 1.获取tbale
def get_table():
    return None


# 2.paddle ocr文本识别
def det_table():
    ocr_data = ocr.ocr(img)
    txts = [x[1][0] for x in ocr_data]
    bboxes = [x[0] for x in ocr_data]
    return bboxes, txts


# 3.获取master 结果
def get_master_result():
    structure_master_config = '/home/zhaohj/Documents/workspace/github_my/TableMASTER-mmocr/configs/textrecog/master/table_master_ResnetExtract_Ranger_0705.py'
    structure_master_ckpt = '/home/zhaohj/Documents/workspace/github_my/TableMASTER-mmocr/output/epoch_97.pth'
    fpath = '/home/zhaohj/Documents/dataset/Table/TAL/val/1cb58c2819530bf7faf03526518197c8.jpg'
    master_structure_inference = Structure_Recognition(structure_master_config,
                                                       structure_master_ckpt)
    _, result_dict = master_structure_inference.predict_single_file(fpath)
    return result_dict
