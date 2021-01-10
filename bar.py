import pandas as pd
import os
import configparser
from utils import *

artifact = {'name': '翠绿猎人的容器', 'set_pieces': '空之杯', 'main_stat': '风元素伤害加成', 'main_stat_value': '7.0%', 'star': 5,
            'lv': 0, 'vice_stat0': '暴击伤害', 'vice_stat0_value': '6.2%', 'vice_stat1': '生命值', 'vice_stat1_value': '239',
            'vice_stat2': '元素精通', 'vice_stat2_value': '16', 'set_name': '翠绿之影'}
config = configparser.ConfigParser()
config.read('config.ini', encoding='UTF-8')
save_path = str(config['common']['save_path'])

full_path = os.path.join(save_path, '1.xlsx')
if not os.path.exists(full_path):
    dic1 = {'圣遗物名称': [],
            '圣遗物类型': [],
            '主属性': [],
            '主属性数值': [],
            '星级': [],
            '等级': [],
            '副属性1': [],
            '副属性1数值': [],
            '副属性2': [],
            '副属性2数值': [],
            '副属性3': [],
            '副属性3数值': [],
            '副属性4': [],
            '副属性4数值': [],
            '所属套装': [],
            '创建时间': []
            }
    df = pd.DataFrame(dic1)
    df.to_excel(full_path, index=False)

