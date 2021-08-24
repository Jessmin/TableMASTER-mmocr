import numpy as np
import cv2


def run(img):
    h, w, c = img.shape
    _max = max(h, w)
    _min = min(h, w)
    print(_max, _min)
    if _max > 3840 or _min > 2160:
        scale_x = 3840 / _max
        scale_y = 2160 / _min
        print(scale_x, scale_y)
        scale = min(scale_x, scale_y)
        h = int(h * scale)
        w = int(w * scale)
        print(h, w)
        img = cv2.resize(img, (w, h))
    print(img.shape)
    # cv2.imwrite('temp.png', img)


img = np.zeros((3024,4032,3),dtype=np.uint8)
# img = cv2.imread("/home/zhaohj/Pictures/1629707780_617949638.webp")
run(img)