import cv2
import numpy as np


linepoints = np.array([1, 2, 3, 4])
def drawlines(img, lines):
    i = 0
    for line in lines:
        if i == 2:
            for x1, y1, x2, y2 in line:
                linepoints[0] = x1
                linepoints[1] = y1
                linepoints[2] = x2
                linepoints[3] = y2
        else:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=4)
        i += 1
    return img


def draw_lines(img, linez):  # inferior than the og drawlines
    left_lane_lines = []  # slope and intercept of the line
    right_lane_lines = []
    left_weights = []  # length of the line
    right_weights = []
    for points in linez:
        x1, y1, x2, y2 = points[0]
        if x2 == x1:  # slope of undefined
            continue
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  # implementing polyfit to identify slope and intercept
        slope = parameters[0]
        intercept = parameters[1]
        length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        if slope < 0:
            left_lane_lines.append([slope, intercept])
            left_weights.append(length)
        else:
            right_lane_lines.append([slope, intercept])
            right_weights.append(length)
    # Computing average slope and intercept
    left_average_line = np.average(left_lane_lines, axis=0)  # computes a weighted average
    right_average_line = np.average(right_lane_lines, axis=0)
    # print(left_average_line, right_average_line)
    # #Computing weighted sum
    # if len(left_weights)>0:
    #     left_average_line = np.dot(left_weights,left_lane_lines)/np.sum(left_weights)
    # if len(right_weights)>0:
    #     right_average_line = np.dot(right_weights,right_lane_lines)/np.sum(right_weights)
    left_fit_points = get_coordinates(img, left_average_line)
    right_fit_points = get_coordinates(img, right_average_line)
    # print(left_fit_points, right_fit_points)
    return [[left_fit_points], [right_fit_points]]  # returning the final coordinates


def get_coordinates(img, line_parameters):  # functions for getting final coordinates
    print(line_parameters)
    slope = line_parameters[0]
    intercept = line_parameters[1]

    y1 = img.shape[0]
    y2 = img.shape[0] * 0.6  # this value may vary depending on size of image being read
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, int(y1), x2, int(y2)]


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def process(image, roi):
    # a trapezoid specific to this image

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Canny is an edge detection algorithm. Second and Third params are min and max
    canny_image = cv2.Canny(gray_img, 0, 255)

    cropped_img = region_of_interest(canny_image,
                                     np.array([roi], np.int32))  # using function and defined roi to get a trapezoid

    lines = cv2.HoughLinesP(cropped_img, rho=0.5, theta=np.pi / 360, threshold=50, lines=np.array([]), minLineLength=10,
                            maxLineGap=5)
    # Uses r = xcosθ + ysinθ to find possible points of a line in the image. r is a function and of θ and
    # can be graphed as a function. Many points are graphed on the same graph, and the if they interesect at a point
    # the line exists at the point of intersection.
    # theta is just the step value of the angle from 0 to π, rho is the step value of the distance of the line, so if rho=1, it steps by 1 pixel
    # having smaller values makes the lines drawn mega accurate
    return lines


def find_angle(linepoints):
    slope = abs(linepoints[3] - linepoints[1]) / abs(linepoints[2] - linepoints[0])
    print(slope)
    theta = np.arctan(slope) * 180 / np.pi
    return theta


def draw(frame, lines, line_points):
    frame = drawlines(frame, lines)
    angle = find_angle(linepoints=line_points)
    cv2.line(frame, (int(linepoints[0]), int(line_points[1])), (int(line_points[2]), int(line_points[3])),
             color=(255, 255, 255), thickness=10)
    cv2.line(frame, (int(line_points[0]), int(line_points[1])), (int(line_points[0] + 100), int(line_points[1])),
             color=(255, 0, 0), thickness=10)
    cv2.putText(frame, f" Angle = {angle}", org=(50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6,
                color=(0, 255, 0), thickness=2)
    return frame, angle


def stay_in_line(img):
    c = img.copy()
    height = img.shape[0]
    width = img.shape[1]
    angle = 0
    roi_left = [(50, height - 20),
                (width / 2 - 110, height / 2 + 100),
                (width / 2, height / 2 + 100),
                (width / 2, height - 20)]
    roi_right = [(600, height - 20),
                 (width / 2 + 110, height / 2 + 100),
                 (width / 2, height / 2 + 100),
                 (width / 2, height - 20)]

    lines_left = process(img, roi_left)
    lines_right = process(img, roi_right)
    if lines_left is not None and lines_right is None:
        img, angle = draw(img, lines_left, linepoints)
    if lines_right is not None and lines_left is None:
        img, angle = draw(img, lines_right, linepoints)
    if lines_right is not None and lines_right is not None:
        if len(lines_right) > len(lines_left):
            img, angle = draw(img, lines_right, linepoints)
        elif len(lines_left) > len(lines_right):
            img, angle = draw(img, lines_left, linepoints)

    cv2.imshow('FRAME', img)
    cv2.waitKey(1)
    return angle, img


if __name__ == '__main__':
    path = input("Video Path: ")
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        stay_in_line(frame)
