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

img_path = os.path.join(path, os.listdir(path)[-1])
print(img_path)
img = cv2_imread(img_path)
img = cv2.resize(img, (1920, 1080))

img = img_crop(img)

cv2.imshow("houghline", img)
cv2.waitKey()
cv2.destroyAllWindows()

print(get_stat(img, access_token))
