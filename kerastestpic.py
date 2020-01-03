from keras import models
from keras.models import load_model
import cv2

from keras.preprocessing import image

import numpy as np

# 加载预训练模型

model = models.load_model("model/tongue_99.h5")

# 准备测试图片各一张

imgpath = "test03.jpg"

img = cv2.imread(imgpath)
img = cv2.resize(img, (200, 200))
# RGB图像转为gray
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# 需要用reshape定义出图片的 通道数，图片的长与宽
# img_data = (img.reshape( 200, 200, 1))
pre_x = (img.reshape(1, 200, 200, 1))

# 分类
r = model.predict(pre_x)
print("result = ")
print(r)

