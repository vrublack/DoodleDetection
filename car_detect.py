# import the necessary packages
import numpy as np
import argparse
import cv2
import os

# opencv uses BGR
import sys

wheel_color = (171, 186, 211)[::-1]
body_color = (255, 0, 0)[::-1]


def contour_center(contour):
    M2 = cv2.moments(contour)
    return (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))


def resize_contour(contour, factor):
    M = cv2.moments(contour)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    scaled_contour = np.copy(contour)

    for i in range(scaled_contour.shape[0]):
        scaled_contour[i][0][0] = int(contour[i][0][0] * factor)
        scaled_contour[i][0][1] = int(contour[i][0][1] * factor)

    M2 = cv2.moments(scaled_contour)
    new_center = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))

    # maintains center
    for i in range(scaled_contour.shape[0]):
        scaled_contour[i][0][0] -= new_center[0] - center[0]
        scaled_contour[i][0][1] -= new_center[1] - center[1]

    return scaled_contour


def add_alpha(img):
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255  # creating a dummy alpha channel image.
    img_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_RGBA


def fill(img, contour, color):
    corners = np.zeros((1, contour.shape[0], 2), dtype=np.int32)
    for i in range(contour.shape[0]):
        corners[0][i][0] = contour[i][0][0]
        corners[0][i][1] = contour[i][0][1]
    cv2.fillPoly(img, corners, color)


def crop(image, contour, output_fname):
    # paint wheel
    center = contour_center(contour)
    flood_fill(image, center, wheel_color)

    image = add_alpha(image)
    mask = np.zeros(image.shape, dtype=np.uint8)

    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
    fill(mask, contour, ignore_mask_color)

    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)

    # save the result
    cv2.imwrite(output_fname, masked_image)
    with open(output_fname + '_.info', 'w') as info:
        info.write('{}x{}'.format(center[0], center[1]))


def flood_fill(img, seed, color):
    blank_image = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(img, blank_image, seed, color)


def extract_wheels(image, circles, output_index=0):
    image = image.copy()
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[:]).astype("int")
    contour_list = []
    radius = max(int(circles[0][2] * 2), int(circles[1][2] * 1.3))
    cropped_images = []
    centers = []
    approxims = []
    # loop over the (x, y) coordinates and radius of the circles
    for j, (x, y, r) in enumerate(circles):
        # crop image
        cropped_image = image[y - radius:y + radius, x - radius:x + radius]
        cropped_images.append(cropped_image)
        bilateral_filtered_image = cv2.bilateralFilter(cropped_image, 5, 175, 175)
        edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
        _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.imshow("output", cropped_image)
        # cv2.waitKey(0)

        # just take contour with greatest area
        max_approx = (None, -1)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.001 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(approx)

            if area > max_approx[1]:
                max_approx = (approx, area)

        approx = max_approx[0]
        approx = resize_contour(approx, 1.1)
        contour_list.append(approx)
        crop(cropped_image, approx, 'wheel_{}_{}.png'.format(j, output_index))

        for i in range(approx.shape[0]):
            approx[i, 0, 0] += x - radius
            approx[i, 0, 1] += y - radius
        # fill(image, approx, (255, 255, 255))
        center = contour_center(approx)
        centers.append(center)

        approxims.append(approx)

    # paint car body
    # estimate body center position based on wheel proportions
    x_dist = abs(centers[0][0] - centers[1][0])
    x_avg = centers[0][0] / 2 + centers[1][0] / 2
    y_avg = centers[0][1] / 2 + centers[1][1] / 2
    body_center = (x_avg, int(y_avg - x_dist * 0.3))
    flood_fill(image, body_center, body_color)

    # remove wheels from car
    fill(image, approxims[0], (255, 255, 255))
    fill(image, approxims[1], (255, 255, 255))

    cv2.imwrite('car_without_wheels_{}.png'.format(output_index), image)


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


def process_image(image, output_index):
    output = image.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.3, 250, param2=70)

    if circles is None:
        print('No wheels detected!')
        return

    circles = circles[0]

    circles = filter(circles)

    if len(circles) < 2:
        print('Fewer than two circles detected!')
        return

    extract_wheels(image, circles, output_index)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('This script takes exactly one argument, which is the input file!')
        exit(1)
    input_fname = sys.argv[1]
    process_image(cv2.imread(input_fname), 0)
