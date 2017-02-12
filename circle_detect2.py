# import the necessary packages
import numpy as np
import argparse
import cv2
import os


def graph(orig_image, image, circles):
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[:]).astype("int")

        radius = max(int(circles[0][2] * 1.3), int(circles[1][2] * 1.3))
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles: 
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(image, (x, y), radius, (0, 255, 0), 4)
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # show the output image
        cv2.imshow("output", np.hstack([orig_image, image]))
        cv2.waitKey(0)


def check_circle(c, box):
    # is center of circle in box?
    return box[0] <= c[0] <= box[2] and box[1] <= c[1] <= box[3]


def check_model_output(circles, label):
    if circles is None or len(circles) != 2:
        return False

    return check_circle(circles[0], label[0]) and check_circle(circles[1], label[1]) \
           or check_circle(circles[0], label[1]) and check_circle(circles[1], label[0])


def filter(circles):
    if len(circles) > 2:
        # filter out circle with higher radius
        circles = sorted(circles, key=lambda x: x[2])
        # only return 2 smallest circles
        return circles[:2]
    else:
        return circles


def validate_using_params(images, labels, param1, param2, param3, show_graph):
    correct = 0

    for i, image in enumerate(images):
        output = image.copy()
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        # detect circles in the image
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, param1, param2, param2=param3)

        if circles is None:
            continue

        circles = circles[0]

        circles = filter(circles)

        print(i)

        if show_graph and circles is not None:
            graph(image, output, circles)

        # print(len(circles))

        if check_model_output(circles, labels[i]):
            correct += 1

    return correct


orig_images = []
for f in os.listdir('train'):
    orig_images.append(cv2.imread(os.path.join('train', f)))

orig_labels = [([0, 0, 0, 0], [0, 0, 0, 0]) for _ in orig_images]
with open(os.path.join('labels', 'front-wheels.txt')) as f:
    i = 0
    for l in f.readlines():
        l = l[:-1]
        if len(l) == 0:
            continue
        i += 1
        if l == 'none':
            continue
        coordinates = l.split(',')
        for c in range(4):
            orig_labels[i - 1][0][c] = int(coordinates[c].strip())

with open(os.path.join('labels', 'back-wheels.txt')) as f:
    i = 0
    for l in f.readlines():
        l = l[:-1]
        if len(l) == 0:
            continue
        i += 1
        if l == 'none':
            continue
        coordinates = l.split(',')
        for c in range(4):
            orig_labels[i - 1][1][c] = int(coordinates[c].strip())


images = []
labels = []

# only look at ones that work for now
keep_indexes = [3, 6, 8, 11, 12, 14, 17, 20, 24, 28, 32, 33, 37, 38, 40, 42, 48, 51, 52, 61]
for i in range(len(orig_images)):
    if i in keep_indexes:
        images.append(orig_images[i])
        labels.append(orig_labels[i])


def optimize():
    global images
    global labels

    def drange(start, stop, step):
        r = start
        while r < stop:
            yield r
            r += step

    best_result = -1
    best_params = (-1, -1, -1)

    for p1 in drange(1.1, 1.6, 0.1):
        for p2 in range(250, 600, 50):
            # for p3 in range(40, 150, 10):
            p3 = 70
            print('{}, {}, {}:'.format(p1, p2, p3))
            result = validate_using_params(images, labels, p1, p2, p3, False)
            print(result)
            if result > best_result:
                best_result = result
                best_params = (p1, p2, p3)

    print('Best value {} for params {}, {}, {}'.format(best_result, best_params[0], best_params[1], best_params[2]))


#optimize()

print(validate_using_params(images, labels, 1.3, 250, 70, True))
