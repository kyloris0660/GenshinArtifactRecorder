import cv2
import numpy as np
import os
from math import sqrt
import sys
import requests
import configparser
from utils import *

config = configparser.ConfigParser()
config.read('config.ini', encoding='UTF-8')
path = str(config['common']['screenshot_location'])
access_token = str(config['oath']['access_token'])


def cv2_imread(img_path):
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)


def stat_num_4(result):
    """
    判断副词条数量
    :param result: OCR结果列表
    :return: bool值 0代表3词条，1代表4词条
    """
    set_list = ['幸运儿', '游医', '冒险家', '学士', '战狂', '祭冰之人', '奇迹', '勇士之心', '教官', '祭火之人', '赌徒',
                '祭水之人', '武人', '守护之心', '祭雷之人', '流放者', '行者之心', '炽烈的炎之魔女', '角斗士的终幕礼',
                '如雷的盛怒', '冰风迷途的勇士', '染血的骑士道', '昔日宗室之仪', '沉沦之心', '悠古的磐岩',
                '翠绿之影', '流浪大地的乐团', '逆飞的流星', '平息鸣雷的尊者', '渡过烈火的贤人', '被怜爱的少女']
    loc = [i[:-1] in set_list for i in result].index(True)
    assert loc == 9 or loc == 10
    return loc - 9


img_path = os.path.join(path, os.listdir(path)[-2])
print(img_path)
img = cv2_imread(img_path)
img = cv2.resize(img, (1920, 1080))

img = img_crop(img)

cv2.imshow("houghline", img)
cv2.waitKey()
cv2.destroyAllWindows()


print(get_stat(img, access_token))

