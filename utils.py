import numpy as np
import cv2
import sys
import base64
import requests
from math import sqrt


def cv2_imread(img_path):
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)


def img_crop(img):
    """
    从图片中裁剪圣遗物面板
    :param img: 输入图像向量， 默认(1920, 1080, 3)
    :return:  输出图像向量， 大小约为(810, 490, 3)
    """
    assert img.shape[1] / img.shape[0] == 1920 / 1080

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


def image_to_base64(img_np):
    image = cv2.imencode('.jpg', img_np)[1].tostring()
    return base64.b64encode(image)


class Artifact:
    """
    圣遗物类
    """
    set_pieces = ''
    set_name = ''
    star = 5
    lv = 0
    main_stat = ''
    main_stat_value = ''
    vice_stat0 = ''
    vice_stat0_value = ''
    vice_stat1 = ''
    vice_stat1_value = ''
    vice_stat2 = ''
    vice_stat2_value = ''
    vice_stat3 = ''
    vice_stat3_value = ''

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return str(self.__dict__)


def get_stat(img, access_token):
    img = image_to_base64(img)
    params = {"image": img}
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)

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

    if response:
        result = response.json()['words_result']
        result = [str(i['words']) for i in result]

        artifact = Artifact(result[0])
        artifact.set_pieces = result[1]
        artifact.main_stat = result[2]
        artifact.main_stat_value = int(result[3])
        artifact.star = len(result[4])
        artifact.lv = int(result[5])
        artifact.vice_stat0 = result[6].split('+')[0][1:]
        artifact.vice_stat0_value = result[6].split('+')[1]
        artifact.vice_stat1 = result[7].split('+')[0][1:]
        artifact.vice_stat1_value = result[7].split('+')[1]
        artifact.vice_stat2 = result[8].split('+')[0][1:]
        artifact.vice_stat2_value = result[8].split('+')[1]
        if stat_num_4(result):
            artifact.vice_stat3 = result[9].split('+')[0][1:]
            artifact.vice_stat3_value = result[9].split('+')[1]
            artifact.set_name = result[10][:-1]
        else:
            artifact.set_name = result[9][:-1]
        return artifact
