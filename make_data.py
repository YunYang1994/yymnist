#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : make_data.py
#   Author      : YunYang1994
#   Created date: 2019-07-12 20:53:30
#   Description :
#
#================================================================

import os
import cv2
import numpy as np
import shutil
import random

if os.path.exists("./Images/"): shutil.rmtree("./Images/")
if os.path.exists("./Annotations"): shutil.rmtree("./Annotations/")
os.mkdir("./Images/")
os.mkdir("./Annotations/")

image_paths  = [os.path.join(os.path.realpath("."), "./mnist/train/" + image_name) for image_name in os.listdir("./mnist/train")]
image_paths += [os.path.join(os.path.realpath("."), "./mnist/test/" + image_name) for image_name in os.listdir("./mnist/test")]

def compute_iou(box1, box2):
    """xmin, ymin, xmax, ymax"""

    A1 = (box1[2] - box1[0])*(box1[3] - box1[1])
    A2 = (box2[2] - box2[0])*(box2[3] - box2[1])

    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    if ymin >= ymax or xmin >= xmax: return 0
    return  ((xmax-xmin) * (ymax - ymin)) / (A1 + A2)


def make_image(data, image_path, ratio=1):

    blank = data[0]
    boxes = data[1]
    label = data[2]

    ID = image_path.split("/")[-1][0]
    image = cv2.imread(image_path)
    image = cv2.resize(image, (int(28*ratio), int(28*ratio)))
    h, w, c = image.shape

    while True:
        xmin = np.random.randint(0, 416-w, 1)[0]
        ymin = np.random.randint(0, 416-h, 1)[0]
        xmax = xmin + w
        ymax = ymin + h
        box = [xmin, ymin, xmax, ymax]

        iou = [compute_iou(box, b) for b in boxes]
        if max(iou) < 0.05:
            boxes.append(box)
            label.append(ID)
            break

    for i in range(w):
        for j in range(h):
            x = xmin + i
            y = ymin + j
            blank[y][x] = image[j][i]

    # cv2.rectangle(blank, (xmin, ymin), (xmax, ymax), [0, 0, 255], 2)
    return blank

with open("Annotations/label.txt", "w") as wf:
    for i in range(2000):
        image_path = os.path.join(os.path.realpath("."), "./Images/%05d.jpg" %i)
        annotation = image_path
        blanks = np.ones(shape=[416, 416, 3]) * 255
        bboxes = [[0,0,1,1]]
        labels = [0]
        data = [blanks, bboxes, labels]

        N = random.randint(0,3)
        for _ in range(N):
            idx = random.randint(0, 54999)
            data[0] = make_image(data, image_paths[idx], 4)

        N = random.randint(0,3)
        for _ in range(N):
            idx = random.randint(0, 54999)
            data[0] = make_image(data, image_paths[idx], 3)

        N = random.randint(0,5)
        for _ in range(N):
            idx = random.randint(0, 54999)
            data[0] = make_image(data, image_paths[idx], 2)

        N = random.randint(5,10)
        for _ in range(N):
            idx = random.randint(0, 54999)
            data[0] = make_image(data, image_paths[idx], 1)

        N = random.randint(0,3)
        for _ in range(N):
            idx = random.randint(0, 54999)
            data[0] = make_image(data, image_paths[idx], 0.5)

        cv2.imwrite(image_path, data[0])
        for i in range(len(labels)):
            if i == 0: continue
            xmin = str(bboxes[i][0])
            ymin = str(bboxes[i][1])
            xmax = str(bboxes[i][2])
            ymax = str(bboxes[i][3])
            class_ind = str(labels[i])
            annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
        print("=> %s" %annotation)
        wf.write(annotation + "\n")

