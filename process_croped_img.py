import configparser
import sys
from utils import *

config = configparser.ConfigParser()
config.read('config.ini', encoding='UTF-8')
path = str(config['common']['screenshot_path'])
access_token = str(config['oath']['access_token'])
save_path = str(config['common']['save_path'])
full_path = os.path.join(save_path, '圣遗物登记表.xlsx')
remove_screenshot = config['common']['remove_screenshot'] == '1'

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
            '得分': [],
            '创建时间': []
            }
    df = pd.DataFrame(dic1)
    df.to_excel(full_path, index=False)
    print('创建新excel表格：{}'.format(full_path))
else:
    print('发现登记表{}，追加新圣遗物'.format(full_path))

img_path = str(sys.argv[1])
if os.path.splitext(img_path)[-1] == ".png" or os.path.splitext(img_path)[-1] == ".jpg":
    img = cv2_imread(img_path)
    assert img.shape[0] > img.shape[1]
    artifact = get_stat(img, access_token, get_create_date(img_path))
    name = artifact.add_to_excel(full_path)
    print('登记新圣遗物：{}，分类：{}，主词条：{}，副词条得分：{}'.format(name, artifact.set_pieces, artifact.main_stat,
                                                   artifact.get_initial_score()))
else:
    print('输入不符合要求')