import csv
import os

import cv2
from PIL import Image

with open('labels/wheels.txt', 'rU') as inputfile:  # whatever he called it
    reader = csv.reader(inputfile)
    dp = list(reader)

directory = 'train'  ##whatever place the png are in


def in_box(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def wheelFinder(square):
    imgs = [cv2.imread(os.path.join(directory, filename)) for filename in os.listdir(directory)]
    generated_i = 0
    for img_i in range(len(imgs)):
        ##iterate over every pixel
        img = imgs[img_i]
        SQUARE = square
        ##grab the corresponding data points of that file ( back and front wheel  fx1, fy1, fx2, fy2, bx1, bx1, bx2, by2) - list dp
        for x in range(0, img.shape[0] - SQUARE, 100):
            for y in range(1, img.shape[1] - SQUARE, 100):
                # all values dp[a][:5] have to be inside of box, or all values of dp[a][5:]

                # dp: x1, y1, x2, y2, ... (same for wheel 2)

                box = (x, y, x + SQUARE, y + SQUARE)
                wheel1_p1 = (int(dp[img_i][0]), int(dp[img_i][1]))
                wheel1_p2 = (int(dp[img_i][2]), int(dp[img_i][3]))
                wheel2_p1 = (int(dp[img_i][4]), int(dp[img_i][5]))
                wheel2_p2 = (int(dp[img_i][6]), int(dp[img_i][7]))

                wheel1_check = in_box(wheel1_p1, box) and in_box(wheel1_p2, box)
                wheel2_check = in_box(wheel2_p1, box) and in_box(wheel2_p2, box)

                cropped = img[y: y + SQUARE, x: x + SQUARE]

                generated_i += 1
                cv2.imwrite('generated_train/file{}_{}.png'.format(generated_i, wheel1_check or wheel2_check), cropped)


#for z in range(300, 400, 50):
wheelFinder(500)
