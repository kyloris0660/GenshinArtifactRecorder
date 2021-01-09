import numpy as np
import cv2
import sys
from math import sqrt


def img_crop(img):
    '''
    从图片中裁剪圣遗物面板
    :param img: 输入图像向量， 默认(1920, 1080, 3)
    :return:  输出图像向量， 大小约为(810, 490, 3)
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 100
    maxLineGap = 50
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    def distance(x1, x2, y1, y2):
        return sqrt(abs((abs(x2 - x1) ** 2 - abs(y2 - y1) ** 2)))

    min_x1 = sys.maxsize
    max_x2 = 0
    min_y1 = sys.maxsize
    max_y2 = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            if distance(x1, x2, y1, y2) > 400:
                min_x1 = min(x1, min_x1)
                max_x2 = max(x2, max_x2)
                min_y1 = min(y1, min_y1)
                max_y2 = max(y2, max_y2)

    return img[min_y1:max_y2, min_x1:max_x2]
