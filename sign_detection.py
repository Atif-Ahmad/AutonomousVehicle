import re
import torch
import cv2
import numpy as np
import keras
from pytesseract import pytesseract
from PIL import Image

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
this_dict = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}


def detect_stop_sign(img, model):
    copy = cv2.imread(img)
    results = model(img)
    results.pandas()  # or .show(), .save(), .crop(), .pandas(), etc
    width = []
    for arr in results.xyxy[0]:
        x1, y1, x2, y2, conf, c = arr
        if int(c) == 11:
            width.append(float(x2) - float(x1))
            cv2.rectangle(copy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0))
            cv2.putText(copy, this_dict.get(int(c)), (int(x1), int(y1)), FONT_FACE, FONT_SCALE, (0, 0, 255), THICKNESS,
                        cv2.LINE_AA)

    cv2.imshow('img', copy)
    cv2.waitKey(0)
    return width


def detect_speed_sign(img, model):
    copy = cv2.imread(img)
    results = model.predict(img)
    results.pandas()  # or .show(), .save(), .crop(), .pandas(), etc
    width = []
    for arr in results.xyxy[0]:
        x1, y1, x2, y2, conf, c = arr
        width.append(float(x2) - float(x1))
        cv2.rectangle(copy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0))
        cv2.putText(copy, this_dict.get(int(c)), (int(x1), int(y1)), FONT_FACE, FONT_SCALE, (0, 0, 255), THICKNESS,
                    cv2.LINE_AA)
    cv2.imshow('img', copy)
    cv2.waitKey(0)


def predict_traffic_light(img):
    labels = ['back', 'go', 'stop', 'warning']
    model = keras.models.load_model('/Users/atif/PycharmProjects/Crossroads/models/traffic_light_v1.h5')
    pred = model.predict(img)
    m = pred[0][0]
    index = 0
    for i in range(1, len(pred[0])):
        if pred[0][i] > m:
            m = pred[0][i]
            index = i

    print(labels[index], '-', m)
    # cv2.putText(img, str(m), (50, 150), FONT_FACE, FONT_SCALE, (0, 0, 255), THICKNESS, cv2.LINE_AA)
    # cv2.imshow('org', img)
    # cv2.waitKey(0)
    return labels[index], m


def find_speed(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, config='--psm 11')
    print(text[:-1])
    return re.findall(r'\d+', text[:-1])


def detect_all_signs(img_path):
    from roboflow import Roboflow
    rf = Roboflow(api_key="bpz1HGZDDIN51H4spjGE")
    project = rf.workspace().project("traffic-light-xdta8")
    model = project.version(1).model

    results = model.predict(img_path, confidence=40, overlap=30).json()

    img = cv2.imread(img_path)
    copy = cv2.imread(img_path)

    for arr in results.get('predictions'):
        x1 = arr.get('x')
        x2 = arr.get('width')
        y1 = arr.get('y')
        y2 = arr.get('height')

        x1 = x1 - x2 / 2
        x2 = x1 + x2
        y1 = y1 - y2 / 2
        y2 = y1 + y2
        c = arr.get('class')
        if c == 'crosswalk':
            print(float(x2) - float(x1), c)
        if c == 'speedlimit':
            # mnist code no need to read width because you gotta slow down/speed up as soon as you see it
            roi = img[int(y1):int(y2), int(x1):int(x2)]
            roi = cv2.resize(roi, (224, 224))
            # speed = find_speed(roi)
            # print(speed)
        if c == 'stop':
            print(float(x2) - float(x1))
        if c == 'trafficlight':  # or pass depending on how well it predicts
            roi = img[int(y1):int(y2), int(x1):int(x2)]
            roi = cv2.resize(roi, (224, 224))
            roi = roi.reshape(1, 224, 224, 3)
            a, b = predict_traffic_light(roi)
            print(a)
        cv2.rectangle(copy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0))
        cv2.putText(copy, c, (int(x1), int(y1)), FONT_FACE, FONT_SCALE, (0, 0, 255), THICKNESS, cv2.LINE_AA)
    return copy


if __name__ == '__main__':
    inp = input("Path: ")
    while inp != 'END':
        copy = detect_all_signs(inp)
        cv2.imshow('img', copy)
        cv2.waitKey(0)
        inp = input("Path: ")
