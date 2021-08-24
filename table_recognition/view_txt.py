import glob
import cv2
import os
import numpy as np

input_dir = '/home/zhaohj/Documents/ThunderDownLoad/pubtabnet/processed_img/StructureLabelAddEmptyBbox_train'
images_dir = '/home/zhaohj/Documents/ThunderDownLoad/pubtabnet/train'
files = glob.glob(f'{input_dir}/*.txt')
for file in files:
    _, fname = os.path.split(file)
    img_fname = os.path.join(images_dir, fname.replace('.txt', '.png'))
    image = cv2.imread(img_fname)
    with open(file, 'r') as f:
        data = f.readlines()
        bboxes = data[2:]
        for box in bboxes:
            box = box.replace('\n','').split(',')
            x, y, w, h = np.int0(box)
            pt1 = (x,y)
            pt2 = (x+w, y+h)
            cv2.rectangle(image, pt1, pt2,(255,255,0),2)
        cv2.imshow('img', image)
        cv2.waitKey(0)
