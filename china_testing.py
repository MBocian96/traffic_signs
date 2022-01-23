import csv
import os

import cv2
from cv2 import imread
import numpy as np
from tensorflow import keras

model = keras.models.load_model('china_model.h5')

model.summary()


def resize_cv(img):
    return cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)


main_folder = 'ChineseTrafficSigns'
dataset = os.path.join(main_folder, 'tsrd-test')
signs = {}
# img_path = "GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/" + image_name
# csv_file = pd.read_csv('ChineseTrafficSigns/index.csv', sep=';')
with open(os.path.join(main_folder, 'test_index.csv'), mode='r') as csv_file:
    counter = 0
    file_len = 1994
    head = next(csv_file)
    for row in csv_file:
        counter += 1
        try:
            row = row.split(';')
            file_name = row[0]
            print(f"{str(counter)}/{str(file_len)}", end="\r")
            img = imread(os.path.join(dataset, str(row[0])))
            class_id = row[7]
            roix1 = int(row[3])
            roiy1 = int(row[4])
            roix2 = int(row[5])
            roiy2 = int(row[6])
            img = img[roix1:roix2, roiy1:roiy2, :]
        except TypeError as err:
            print(err)
            print(file_name)
            continue
        img = resize_cv(img)
        stack_img = np.stack([img])
        pred = model.predict([stack_img, ])
        predicted_class_id = np.argmax(pred[0])
        try:
            result_table = signs[str(class_id)]
        except KeyError:
            signs.update({
                str(class_id): {
                    'ok': 0,
                    'not': 0,
                }
            })
            result_table = signs[str(class_id)]
        key = 'not'
        if predicted_class_id == class_id:
            key = 'ok'
        how_much = result_table[key] + 1
        result_table[key] = how_much

with open("chinese_test.csv", 'w') as test_csv:
    csv_writer = csv.writer(test_csv, delimiter=',')
    csv_writer.writerow(["ClassID", "OK", "NOT"])
    for sign_id, results in signs.items():
        csv_writer.writerow([str(sign_id), results['ok'], results['not']])
