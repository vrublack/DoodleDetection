# import the necessary packages
import numpy as np
import argparse
import cv2
import os

def check_circle(c, box):
    # is center of circle in box?
    return box[0] <= c[0] <= box[2] and box[1] <= c[1] <= box[3]

def check_model_output(circles, label):
    if circles is None or len(circles[0]) != 2:
        return False

    return check_circle(circles[0][0], label[0]) and check_circle(circles[0][1], label[1]) \
           or check_circle(circles[0][0], label[1]) and check_circle(circles[0][1], label[0])

def validate_using_params(images, labels, param1, param2, param3):
    correct = 0

    for i, image in enumerate(images):
        output = image.copy()
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        # detect circles in the image
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, param1, param2, param2=param3)

        #print(len(circles[0]))

        if check_model_output(circles, labels[i]):
            correct += 1

    return correct


images = []
for f in os.listdir('train'):
    images.append(cv2.imread(os.path.join('train', f)))

labels = [([0, 0, 0, 0], [0, 0, 0, 0]) for _ in images]
with open(os.path.join('labels', 'front-wheels.txt')) as f:
    i = 0
    for l in f.readlines():
        l = l[:-1]
        i += 1
        if len(l) == 0:
            continue
        coordinates = l.split(',')
        for c in range(4):
            labels[i - 1][0][c] = int(coordinates[c].strip())

with open(os.path.join('labels', 'back-wheels.txt')) as f:
    i = 0
    for l in f.readlines():
        l = l[:-1]
        i += 1
        if len(l) == 0:
            continue
        coordinates = l.split(',')
        for c in range(4):
            labels[i - 1][1][c] = int(coordinates[c].strip())

# labels have only been written up to this point
images = images[:43]
labels = labels[:43]

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


for p1 in drange(1.1, 1.6, 0.1):
    for p2 in range(250, 600, 50):
        #for p3 in range(40, 150, 10):
        p3 = 200
        print('{}, {}, {}:'.format(p1, p2, p3))
        print(validate_using_params(images, labels, p1, p2, 70))


