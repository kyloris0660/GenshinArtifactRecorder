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

    def get_set_list_index(result):
        """
        判断圣遗物套装所在位置
        :param result: OCR结果列表
        :return: int 表示位置
        """
        set_list = ['幸运儿', '游医', '冒险家', '学士', '战狂', '祭冰之人', '奇迹', '勇士之心', '教官', '祭火之人', '赌徒',
                    '祭水之人', '武人', '守护之心', '祭雷之人', '流放者', '行者之心', '炽烈的炎之魔女', '角斗士的终幕礼',
                    '如雷的盛怒', '冰风迷途的勇士', '染血的骑士道', '昔日宗室之仪', '沉沦之心', '悠古的磐岩',
                    '翠绿之影', '流浪大地的乐团', '逆飞的流星', '平息鸣雷的尊者', '渡过烈火的贤人', '被怜爱的少女']
        loc = [i[:-1] in set_list for i in result].index(True)
        return loc

    def get_index(lst, item):
        return [index for (index, value) in enumerate(lst) if value == item]

    def get_vice_stat_index(result):
        """
        判断圣遗物副词条所在位置
        :param result: OCR结果列表
        :return: int 表示位置
        """
        stat_list = ['攻击力', '生命值', '防御力', '元素精通', '元素充能效率', '暴击率', '暴击伤害', '风元素伤害加成', '火元素伤害加成',
                     '水元素伤害加成', '雷元素伤害加成', '冰元素伤害加成', '岩元素伤害加成' '物理伤害加成']
        loc = get_index([i.split('+')[0] in stat_list for i in result], True)[1:]
        return loc

    if response:
        print(response.json())
        result = response.json()['words_result']
        result = [str(i['words']).replace('·', '') for i in result]

        artifact = Artifact(result[0])
        artifact.set_pieces = result[1]
        artifact.main_stat = result[2]
        artifact.main_stat_value = result[3]
        artifact.star = len(result[4])
        artifact.lv = int(result[5])

        vice_stat_index = get_vice_stat_index(result)
        set_list_index = get_set_list_index(result)
        artifact.vice_stat0 = result[vice_stat_index[0]].split('+')[0]
        artifact.vice_stat0_value = result[vice_stat_index[0]].split('+')[1]
        artifact.vice_stat1 = result[vice_stat_index[1]].split('+')[0]
        artifact.vice_stat1_value = result[vice_stat_index[1]].split('+')[1]
        artifact.vice_stat2 = result[vice_stat_index[2]].split('+')[0]
        artifact.vice_stat2_value = result[vice_stat_index[2]].split('+')[1]
        if len(vice_stat_index) == 4:
            artifact.vice_stat3 = result[vice_stat_index[3]].split('+')[0]
            artifact.vice_stat3_value = result[vice_stat_index[3]].split('+')[1]
        artifact.set_name = result[set_list_index][:-1]

        return artifact
