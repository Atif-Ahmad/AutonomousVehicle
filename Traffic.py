import torch
import cv2
from keras.models import load_model

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

path_to_yolo = input("Path to yolov5 repo: ")
model = torch.hub.load(path_to_yolo, 'yolov5s', source='local')
keras_model = load_model('model_v1.h5')
# Image
inp = input("Path: ")
while inp != 'END':
    img = cv2.imread(inp)
    labels = ['go', 'goForward', 'goLeft', 'stopLeft', 'stop', 'warning', 'warningLeft'] #a could be green
    copy = img.copy()
    # Inference
    results = model(img)
    # Results, change the flowing to: results.show()
    results.pandas()  # or .show(), .save(), .crop(), .pandas(), etc

    print(results.xyxy[0])

    for arr in results.xyxy[0]:
        x1, y1, x2, y2, conf, c = arr
        print(float(x1))
        if conf > 0.5:
            cv2.rectangle(copy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0))
            cv2.putText(copy, this_dict.get(int(c)), (int(x1), int(y1)), FONT_FACE, FONT_SCALE, (0, 0, 255), THICKNESS,
                        cv2.LINE_AA)

        if (int(c) == 9):
            height, width = img.shape[:2]
            roi = img[int(y1):int(y2), int(x1):int(x2)]
            org = roi.copy()
            org = cv2.resize(org, (224, 224))
            roi = cv2.resize(roi, (224, 224))
            roi = roi.reshape(1, 224, 224, 3)
            pred = keras_model.predict(roi)
            print(pred)

            m = pred[0][0]
            index = 0
            for i in range(1, len(pred[0])):
                if pred[0][i] > m:
                    m = pred[0][i]
                    index = i

            print(labels[index], '-', m)
            cv2.putText(org, str(m), (50, 150), FONT_FACE, FONT_SCALE, (0, 0, 255), THICKNESS, cv2.LINE_AA)
            cv2.imshow('org', org)
            cv2.waitKey(0)

    cv2.imshow("frame", copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    inp = input("Path:")
