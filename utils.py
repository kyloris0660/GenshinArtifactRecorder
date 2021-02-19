import numpy as np
from fuzzywuzzy.fuzz import partial_ratio
from skimage.metrics import structural_similarity
import cv2
import sys
import base64
import requests
from math import sqrt
import os
import pandas as pd


def get_create_date(file):
    """
    获取圣遗物获取日期
    :param file: 图像路径
    :return: 日期，如：2021/1/11
    """
    import time
    # os.stat return properties of a file
    tmpTime = time.localtime(os.stat(file).st_ctime)
    return time.strftime('%Y/%m/%d', tmpTime)


def cv2_imread(img_path):
    """
    中文路径支持
    """
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)


def save_processed_file(path):
    processed_item = [i for i in os.listdir(path)]
    with open('processed_file.txt', 'w') as f:
        for i in processed_item:
            f.write(i)
            f.write('\n')


def add_processed_file(file_name):
    with open('processed_file.txt', 'a') as f:
        f.write(file_name)
        f.write('\n')


def add_ignored_file(file_name):
    with open('ignored_file.txt', 'a') as f:
        f.write(file_name)
        f.write('\n')


def load_processed_file():
    processed_item = []
    for line in open('processed_file.txt', 'r'):
        processed_item.append(line.strip('\n'))
    for line in open('ignored_file.txt', 'r'):
        processed_item.append(line.strip('\n'))
    return processed_item


def img_crop(img):
    """
    从图片中裁剪圣遗物面板
    *使用边缘检测后的顶点坐标*
    :param img: 输入图像向量， 默认(1920, 1080, 3)
    :return:  输出图像向量， 大小约为(810, 490, 3)， 状态位，1为有效
    """
    # assert img.shape[1] / img.shape[0] == 1920 / 1080

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 100
    maxLineGap = 50
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    def distance(x1, x2, y1, y2):
        return sqrt(abs((abs(x2 - x1) ** 2 - abs(y2 - y1) ** 2)))

    x_min = sys.maxsize
    x_max = 0
    y_min = sys.maxsize
    y_max = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            if distance(x1, x2, y1, y2) > 320:
                x_min = min(x_min, x1, x2)
                x_max = max(x_max, x1, x2)
                y_min = min(y_min, y1, y2)
                y_max = max(y_max, y1, y2)

    y_len = y_max - y_min
    x_len = x_max - x_min
    if y_len > 500 > x_len > 480:
        return img[y_min:y_max, x_min:x_max], 1
    else:
        # print(y_len)
        # print(x_len)
        # cv2.imshow('test', img[y_min:y_max, x_min:x_max])
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return img, 0


def img_crop_2(img):
    """
    从图片中裁剪圣遗物面板
    *使用边缘检测宽度，如宽度符合条件而高度不符合则使用固定高度*
    :param img: 输入图像向量， 默认(1920, 1080, 3)
    :return:  输出图像向量， 大小约为(810, 490, 3)， 状态位，1为有效
    """
    # assert img.shape[1] / img.shape[0] == 1920 / 1080

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 100
    maxLineGap = 50
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    def distance(x1, x2, y1, y2):
        return sqrt(abs((abs(x2 - x1) ** 2 - abs(y2 - y1) ** 2)))

    x_min = sys.maxsize
    x_max = 0
    y_min = sys.maxsize
    y_max = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            if distance(x1, x2, y1, y2) > 320:
                x_min = min(x_min, x1, x2)
                x_max = max(x_max, x1, x2)
                y_min = min(y_min, y1, y2)
                y_max = max(y_max, y1, y2)

    y_len = y_max - y_min
    x_len = x_max - x_min
    if 500 > x_len > 480:
        if y_len > 500:
            return img[y_min:y_max, x_min:x_max], 1
        else:
            # cv2.imshow('test', img[y_min:y_min+800, x_min:x_max])
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            return img[y_min:y_min + 800, x_min:x_max], 1
    else:
        return img, 0


def img_crop_3(img):
    """
    从图片中裁剪圣遗物面板
    *16：9屏幕特攻，直接匹配像素值*
    :param img: 输入图像向量， 限定为(1920, 1080, 3)
    :return:  输出图像向量&状态位，1为有效
    """
    feature_img = cv2_imread('FeatureImg.png')
    if structural_similarity(img[1010:1030, 1600:1750], feature_img, multichannel=True) >= 0.99:
        return img[100:800, 1290:1780], 1
    else:
        test_area = img[510:620, 1060:1160]
        unique = list(np.unique(test_area.reshape(-1, 3), axis=0, return_counts=False)[0])
        if unique == [216, 229, 236]:
            return img[110:800, 714:1200], 1
        else:
            return img, 0


def image_to_base64(img_np):
    image = cv2.imencode('.jpg', img_np)[1].tostring()
    return base64.b64encode(image)


def compute_initial_score(stat, value):
    percentage_atk = ['4.1%', '4.7%', '5.3%', '5.8%']
    value_atk = ['14', '16', '18', '19']
    crit_rate = ['2.7%', '3.1%', '3.5%', '3.9%']
    crit_dmg = ['5.4%', '6.2%', '7.0%', '7.8%']
    recharge_rate = ['4.5%', '5.2%', '5.8%', '6.5%']
    if stat == '暴击率':
        if value in crit_rate:
            return crit_rate.index(value) * 0.4 + 2
    elif stat == '暴击伤害':
        if value in crit_dmg:
            return crit_dmg.index(value) * 0.4 + 2
    elif stat == '攻击力':
        if value in percentage_atk:
            return percentage_atk.index(value) * 0.2 + 1
        if value in value_atk:
            return value_atk.index(value) * 0.1 + 0.5
    elif stat == '元素充能效率':
        if value in recharge_rate:
            return recharge_rate.index(value) * 0.2 + 1
    return 0


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
    date = ''

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return str(self.__dict__)

    def get_initial_score(self):
        score = 0
        score += compute_initial_score(self.vice_stat0, self.vice_stat0_value)
        score += compute_initial_score(self.vice_stat1, self.vice_stat1_value)
        score += compute_initial_score(self.vice_stat2, self.vice_stat2_value)
        score += compute_initial_score(self.vice_stat3, self.vice_stat3_value)
        return round(score, 2)

    def add_to_excel(self, path):
        sheet = pd.read_excel(path)
        insert_column = {'圣遗物名称': self.name,
                         '圣遗物类型': self.set_pieces,
                         '主属性': self.main_stat,
                         '主属性数值': self.main_stat_value,
                         '星级': self.star,
                         '等级': self.lv,
                         '副属性1': self.vice_stat0,
                         '副属性1数值': self.vice_stat0_value,
                         '副属性2': self.vice_stat1,
                         '副属性2数值': self.vice_stat1_value,
                         '副属性3': self.vice_stat2,
                         '副属性3数值': self.vice_stat2_value,
                         '副属性4': self.vice_stat3,
                         '副属性4数值': self.vice_stat3_value,
                         '所属套装': self.set_name,
                         '得分': self.get_initial_score(),
                         '创建时间': self.date
                         }
        sheet = sheet.append(insert_column, ignore_index=True)
        sheet.to_excel(path, index=False)
        return self.name


def get_stat(img, access_token, date):
    """
    OCR获取数据写入到圣遗物对象
    :param img: cv2图像对象
    :param access_token: 百度ai的文字识别token
    :param date: 圣遗物获取日期
    :return: 圣遗物对象
    """
    img = image_to_base64(img)
    params = {"image": img}
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
    # request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)

    def get_set_list_index(result):
        """
        判断圣遗物套装所在位置，弃用
        :param result: OCR结果列表
        :return: int 表示位置
        """
        set_list = ['幸运儿', '游医', '冒险家', '学士', '战狂', '祭冰之人', '奇迹', '勇士之心', '教官', '祭火之人', '赌徒',
                    '祭水之人', '武人', '守护之心', '祭雷之人', '流放者', '行者之心', '炽烈的炎之魔女', '角斗士的终幕礼',
                    '如雷的盛怒', '冰风迷途的勇士', '染血的骑士道', '昔日宗室之仪', '沉沦之心', '悠古的磐岩',
                    '翠绿之影', '流浪大地的乐团', '逆飞的流星', '平息鸣雷的尊者', '渡过烈火的贤人', '被怜爱的少女']
        # loc = [i in set_list for i in result].index(True)
        for i in result:
            for j in set_list:
                if partial_ratio(i, j) > 80:
                    return result.index(i)

    def get_set_name(result):
        set_list = ['幸运儿', '游医', '冒险家', '学士', '战狂', '祭冰之人', '奇迹', '勇士之心', '教官', '祭火之人', '赌徒',
                    '祭水之人', '武人', '守护之心', '祭雷之人', '流放者', '行者之心', '炽烈的炎之魔女', '角斗士的终幕礼',
                    '如雷的盛怒', '冰风迷途的勇士', '染血的骑士道', '昔日宗室之仪', '沉沦之心', '悠古的磐岩',
                    '翠绿之影', '流浪大地的乐团', '逆飞的流星', '平息鸣雷的尊者', '渡过烈火的贤人', '被怜爱的少女']
        max = 0
        output = ''
        for item in result:
            for set in set_list:
                ratio = partial_ratio(set, item)
                if ratio > max:
                    max = ratio
                    output = set
        return output

    def get_index(lst, item):
        return [index for (index, value) in enumerate(lst) if value == item]

    def get_kind(kind):
        kind_list = ["生之花", "死之羽", "时之沙", "空之杯", "理之冠"]
        for set_kind in kind_list:
            if partial_ratio(set_kind, kind) > 80:
                return set_kind

    def get_name(kind, name):
        kind_list = ["生之花", "死之羽", "时之沙", "空之杯", "理之冠"]
        name_list = {
            "生之花": ["饰金胸花", "历经风雪的思念", "磐陀裂生之花", "夏祭之花", "游医的银莲", "赌徒的胸花", "学士的书签", "染血的铁之心", "勇士的勋章", "武人的红花",
                    "角斗士的留恋", "教官的胸花", "宗室之花", "守护之花",
                    "野花记忆的绿野", "流放者之花", "渡火者的决绝", "乐团的晨光", "远方的少女之心", "魔女的炎之花", "雷鸟的怜悯", "战狂的蔷薇", "平雷之心", "奇迹之花",
                    "故人之心", "幸运儿绿花", "冒险家之花"],
            "死之羽": ["摧冰而行的执望", "追忆之风", "嵯峨群峰之翼", "夏祭终末", "游医的枭羽", "赌徒的羽饰", "学士的羽笔", "染血的黑之羽", "流放者之羽", "琴师的箭羽", "守护徽印",
                    "勇士的期许", "武人的羽饰", "宗室之翎",
                    "角斗士的归宿", "猎人青翠的箭羽", "少女飘摇的思念", "魔女常燃之羽", "教官的羽饰", "雷灾的孑遗", "渡火者的解脱", "战狂的翎羽", "平雷之羽", "奇迹之羽",
                    "归乡之羽", "幸运儿鹰羽", "冒险家尾羽"],
            "时之沙": ["坚铜罗盘", "冰雪故园的终期", "星罗圭璧之晷", "夏祭之刻", "游医的怀钟", "赌徒的怀表", "学士的时钟", "骑士染血之时", "流放者怀表", "教官的怀表", "终幕的时计",
                    "勇士的坚毅", "守护座钟", "武人的水漏",
                    "角斗士的希冀", "翠绿猎人的笃定", "渡火者的煎熬", "宗室时计", "少女苦短的良辰", "魔女破灭之时", "雷霆的时计", "战狂的时计", "平雷之刻", "奇迹之沙",
                    "逐光之石", "幸运儿沙漏", "冒险家怀表"],
            "空之杯": ["沉波之盏", "遍结寒霜的傲骨", "巉岩琢塑之樽", "夏祭水玉", "游医的药壶", "赌徒的骰盅", "学士的墨杯", "染血骑士之杯", "流放者之杯", "守护之皿", "吟游者之壶",
                    "勇士的壮行", "角斗士的酣醉", "宗室银瓮",
                    "翠绿猎人的容器", "战狂的骨杯", "少女片刻的闲暇", "魔女的心之火", "教官的茶杯", "降雷的凶兆", "渡火者的醒悟", "平雷之器", "武人的酒杯", "奇迹之杯",
                    "异国之盏", "幸运儿之杯", "冒险家金杯"],
            "理之冠": ["酒渍船帽", "破冰踏雪的回音", "不动玄石之相", "夏祭之面", "祭雷礼冠", "祭火礼冠", "祭水礼冠", "祭冰礼冠", "游医的方巾", "学士的镜片", "赌徒的耳环",
                    "染血的铁假面", "渡火者的智慧", "流放者头冠", "宗室面具", "勇士的冠冕",
                    "守护束带", "武人的头巾", "角斗士的凯旋", "翠绿的猎人之冠", "指挥的礼帽", "焦灼的魔女帽", "教官的帽子", "唤雷的头冠", "战狂的鬼面", "平雷之冠",
                    "少女易逝的芳颜", "奇迹耳坠", "感别之冠", "幸运儿银冠", "冒险家头带"]
        }
        out_name = ''
        max = 0
        for set_name in name_list[kind]:
            ratio = partial_ratio(set_name, name)
            if ratio > max:
                max = ratio
                out_name = set_name
        return out_name

    def get_vice_stat_index(result, set_loc):
        """
        判断圣遗物副词条所在位置
        :param result: OCR结果列表
        :return: int 表示位置
        """
        stat_list = ['治疗加成', '攻击力', '生命值', '防御力', '元素精通', '元素充能效率', '暴击率', '暴击伤害', '风元素伤害加成', '火元素伤害加成',
                     '水元素伤害加成', '雷元素伤害加成', '冰元素伤害加成', '岩元素伤害加成' '物理伤害加成']
        # loc = get_index([i.split('+')[0] in stat_list for i in result], True)[1:]
        loc = []
        ptr = 0
        for i in result[:set_loc]:
            for j in stat_list:
                if partial_ratio(i, j) > 95:
                    loc.append(ptr)
                    break
            ptr += 1
        return loc[1:]

    if response:
        result = response.json()['words_result']
        result = [str(i['words']).replace('·', '').replace(':', '') for i in result]
        # print(result)
        artifact = Artifact(result[0])
        artifact.set_pieces = get_kind(result[1])
        artifact.name = get_name(artifact.set_pieces, result[0])
        artifact.main_stat = result[2]
        artifact.main_stat_value = result[3]
        artifact.star = len(result[4])

        if result[5][0] == '+':
            artifact.lv = int(result[5])
        for item in result[5:]:
            if '+' in item:
                if artifact.vice_stat0 == '':
                    artifact.vice_stat0 = item.split('+')[0]
                    artifact.vice_stat0_value = str(item.split('+')[1])
                elif artifact.vice_stat1 == '':
                    artifact.vice_stat1 = item.split('+')[0]
                    artifact.vice_stat1_value = str(item.split('+')[1])
                elif artifact.vice_stat2 == '':
                    artifact.vice_stat2 = item.split('+')[0]
                    artifact.vice_stat2_value = str(item.split('+')[1])
                elif artifact.vice_stat3 == '':
                    artifact.vice_stat3 = item.split('+')[0]
                    artifact.vice_stat3_value = str(item.split('+')[1])

        artifact.set_name = get_set_name(result)
        artifact.date = date

        return artifact
