import os

import cv2
from cv2 import imread
import pandas as pd
from tensorflow import keras

model = keras.models.load_model('model/')

list_images = []
# DATASET = "GTSRB_Final_Test_Images/GTSRB/Final_Test/Images"
DATASET = "ChineseTrafficSigns/"
output = []
for dir in os.listdir(DATASET):
    if dir == '.DS_Store':
        continue
    inner_dir = os.path.join(DATASET, dir)
    csv_file = pd.read_csv(os.path.join(DATASET, 'index.csv'), sep=';')
    for row in csv_file.iterrows():
        img_path = os.path.join(inner_dir, str(row[0]))
        img = imread(img_path)
        img = img[row[1]:row[2], row[3]:row[4], :]
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        list_images.append(img)
        output.append(row[0])
        break
pred = model.predict(img)
print(pred)
