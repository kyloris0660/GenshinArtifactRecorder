import cv2
import numpy as np
import os
from math import sqrt
import sys
from utils import *

path = 'E:/Vedios/Yuan Shen 原神'


def cv2_imread(img_path):
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)


img_path = os.path.join(path, os.listdir(path)[-2])
print(img_path)
img = cv2_imread(img_path)
img = cv2.resize(img, (1920, 1080))

img = img_crop(img)

cv2.imshow("houghline", img)
cv2.waitKey()
cv2.destroyAllWindows()
