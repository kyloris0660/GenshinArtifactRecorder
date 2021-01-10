import configparser
from utils import *

config = configparser.ConfigParser()
config.read('config.ini', encoding='UTF-8')
path = str(config['common']['screenshot_path'])
access_token = str(config['oath']['access_token'])
save_path = str(config['common']['save_path'])
full_path = os.path.join(save_path, '1.xlsx')

for img in os.listdir(path):
    if os.path.splitext(img)[-1] == ".png" or os.path.splitext(img)[-1] == ".jpg":
        img_path = os.path.join(path, img)
        img = cv2_imread(img_path)
        img = cv2.resize(img, (1920, 1080))

        img = img_crop(img)

        artifact = get_stat(img, access_token, get_create_date(img_path))
        artifact.add_to_excel(full_path)

