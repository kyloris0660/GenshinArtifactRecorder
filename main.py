import configparser
import time
from utils import *

config = configparser.ConfigParser()
config.read('config.ini', encoding='UTF-8')
path = str(config['common']['screenshot_path'])
access_token = str(config['oath']['access_token'])
save_path = str(config['common']['save_path'])
full_path = os.path.join(save_path, '圣遗物登记表.xlsx')
remove_screenshot = bool(config['common']['remove_screenshot'])

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
    print('创建新excel表格：{}'.format(full_path))
else:
    print('发现登记表{}，追加新圣遗物'.format(full_path))

processed_item = load_processed_file()
artifact_cnt = 0

for img in os.listdir(path):
    if img not in processed_item:
        if os.path.splitext(img)[-1] == ".png" or os.path.splitext(img)[-1] == ".jpg":
            file_name = img
            img_path = os.path.join(path, img)
            img = cv2_imread(img_path)
            img = cv2.resize(img, (1920, 1080))

            img, have_artifact = img_crop(img)

            if have_artifact:
                time.sleep(0.5)
                artifact = get_stat(img, access_token, get_create_date(img_path))
                name = artifact.add_to_excel(full_path)
                print('登记新圣遗物：{}'.format(name))
                artifact_cnt += 1
                add_processed_file(file_name)
                if remove_screenshot:
                    os.remove(img_path)
            else:
                print('图像{}未检测到有效圣遗物'.format(img_path))

print('共登记{}件新圣遗物'.format(artifact_cnt))
