# Erhu Error Classification System ver. final
# Date created : 11 / 04 / 2022
# Current Progress :
# - Combined extract segmentation with prediction
# - Using newest 5th dataset from professor (10 April 2022)

import os
import sys
import shutil
import cv2
import math
import time

import imutils
import numpy as np
import argparse
import json
# import util.Util as Util
from json import JSONEncoder
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import mediapipe as mp
from math import degrees, atan2
import pixellib
import tensorflow
from pixellib.instance import custom_segmentation
from datetime import datetime
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image, ImageEnhance
from configparser import ConfigParser

sys.path.append(os.path.abspath(os.path.join('..', 'upload')))

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)

tensorflow.config.experimental.enable_tensor_float_32_execution(
    False
)


def getVideoProperties(video):
    properties = {}
    properties['FPS'] = round(video.get(cv2.CAP_PROP_FPS))
    properties['Height'] = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    properties['Width'] = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    properties['Count'] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    return properties


mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles   = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
bow_line = [[0, 0], [0, 0]]
erhu_line = [[0, 0], [0, 0]]

shoulders_left_point = []
shoulders_right_point = []
hip_left_point = []
hip_right_point = []

configure = ConfigParser()
thisfolder = os.path.dirname(os.path.abspath(__file__))
configini = os.path.join(thisfolder, 'config.ini')
configure.read(configini)

K_var = configure.getint('traditional', 'k')
X_var = configure.getint('traditional', 'x')
Y_var = configure.getint('traditional', 'y')
N_var = configure.getint('traditional', 'n')
M_var = configure.getint('traditional', 'm')
P_var = configure.getfloat('deeplearning', 'p')

print("K_var:", K_var)
print("X_var:", X_var)
print("Y_var:", Y_var)
print("N_var:", N_var)
print("M_var:", M_var)
print("P_var:", P_var)

img_chinese = np.zeros((200, 400, 3), np.uint8)
b, g, r, a = 0, 255, 0, 0
chinese_font = './util/simsun.ttc'

result_font = ImageFont.truetype(os.path.join(thisfolder, 'util/simsun.ttc'), 28)
coordTop = []
coordBottom = []
coordLeft = []
coordRight = []
smoothingLeng = 10

def get_contours(frame):
    image = frame.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    return c


def get_angle(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    angles = degrees(atan2(yDiff, xDiff))
    return round(angles, 2)


def angle(a: Point, b: Point, c: Point) -> float:
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def check_orthogonal(x1, y1, x2, y2, x3, y3, x4, y4):
    if (x2 - x1 == 0 and x4 - x3 == 0):
        return False
    elif (x2 - x1 == 0):
        m2 = (y4 - y3) / (x4 - x3)
        if (m2 == 0):
            return True
        else:
            return False
    elif (x4 - x3 == 0):
        m1 = (y2 - y1) / (x2 - x1)
        if (m1 == 0):
            return True
        else:
            return False
    else:
        m1 = (y2 - y1) / (x2 - x1)
        m2 = (y4 - y3) / (x4 - x3)
        if (m1 * m2 == -1):
            return True
        else:
            return False


def erhu_segment(img, frame):
    # determine the most extreme points along the contour
    img_ori = img.copy()
    image = frame.copy()
    # cv2.imshow("Image", img_ori)
    # cv2.waitKey(1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    colorGreen = (0, 255, 0)
    colorRed = (0, 0, 255)
    colorBlue = (255, 0, 0)
    colorYellow = (0, 255, 255)
    colorPurple = (128, 0, 128)
    thickness = 2
    centerPointTop = (extLeft[0] + (extRight[0] - extLeft[0]) // 2, extTop[1])
    centerPointBottom = (extLeft[0] + (extRight[0] - extLeft[0]) // 2, extBot[1])
    centerLengthBottom = centerPointBottom[1] - centerPointTop[1]
    topQinZhenLength = int((centerLengthBottom * 12.2) / 100)
    mark_sectionLength = int((centerLengthBottom * 0.75) / 100)
    bottomQinZhenLength = int((centerLengthBottom * 20.0) / 100)
    middleErhuLength = int((centerLengthBottom * 52.87) / 100)
    # Draw to get the top Qin Zhen
    topQinZhenStart = centerPointTop
    topQinZhenEnd = (centerPointTop[0], centerPointTop[1] + topQinZhenLength)
    rightTopQinZhenDotStart = (extRight[0], extTop[1])
    leftTopQinZhenDotStart = (extLeft[0], extTop[1])
    leftTopQinZhenDotEnd = (extLeft[0], extTop[1] + topQinZhenLength)
    rightTopQinZhenDotEnd = (extRight[0], extTop[1] + topQinZhenLength)
    # cv2.rectangle(img_ori, leftTopQinZhenDotStart, rightTopQinZhenDotEnd, colorPurple, thickness)
    # === Remark for draw all segment of erhu ======================================================================
    mask1 = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask1, leftTopQinZhenDotStart, rightTopQinZhenDotEnd, 255, -1)
    mask_hand1 = cv2.bitwise_and(image, image, mask=mask1)
    # ==============================================================================================================
    # Draw to get the marker line point ============================================================================
    MarkerStart = topQinZhenEnd
    MarkerEnd = (MarkerStart[0], MarkerStart[1] + mark_sectionLength)
    rightMarkerDotStart = rightTopQinZhenDotEnd
    leftMarkerDotStart = leftTopQinZhenDotEnd
    leftMarkerDotEnd = (leftMarkerDotStart[0], leftMarkerDotStart[1] + mark_sectionLength)
    rightMarkerDotEnd = (rightMarkerDotStart[0], rightMarkerDotStart[1] + mark_sectionLength)
    # cv2.rectangle(img_ori, leftTopQinZhenDotStart, rightTopQinZhenDotEnd, colorPurple, thickness)
    # === Remark for draw all segment of erhu ======================================================================
    mask_marker = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask_marker, leftMarkerDotStart, rightMarkerDotEnd, 255, -1)
    mask_markers = cv2.bitwise_and(image, image, mask=mask_marker)
    # ==============================================================================================================
    # === Draw local boundary for Marker Line Point ================================================================
    MarkerLine = mask_markers
    gray = cv2.cvtColor(MarkerLine, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeftMarker = tuple(c[c[:, :, 0].argmin()][0])
    extRightMarker = tuple(c[c[:, :, 0].argmax()][0])
    extTopMarker = tuple(c[c[:, :, 1].argmin()][0])
    extBotMarker = tuple(c[c[:, :, 1].argmax()][0])
    rightTopDot_Marker = (extRightMarker[0], extTopMarker[1])
    leftTopDot_Marker = (extLeftMarker[0], extTopMarker[1])
    leftBottomDot_Marker = (extLeftMarker[0], extBotMarker[1])
    rightBottomDot_Marker = (extRightMarker[0], extBotMarker[1])
    topMiddlePoint_Marker = (
    leftTopDot_Marker[0] + ((rightTopDot_Marker[0] - leftTopDot_Marker[0]) // 2), rightTopDot_Marker[1])
    # ==================================================================================================================
    # Draw to get the bottom QinZhen
    bottomQinZhenStart = MarkerEnd
    bottomQinZhenEnd = (bottomQinZhenStart[0], bottomQinZhenStart[1] + bottomQinZhenLength)
    rightBotQinZhenDotStart = rightMarkerDotEnd
    leftBotQinZhenDotStart = leftMarkerDotEnd
    leftBotQinZhenDotEnd = (leftBotQinZhenDotStart[0], leftBotQinZhenDotStart[1] + bottomQinZhenLength)
    rightBotQinZhenDotEnd = (rightBotQinZhenDotStart[0], rightBotQinZhenDotStart[1] + bottomQinZhenLength)
    # cv2.rectangle(img_ori,leftBotQinZhenDotStart, rightBotQinZhenDotEnd, colorGreen, thickness)
    # === Remark for draw all segment of erhu ======================================================================
    mask2 = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask2, leftBotQinZhenDotStart, rightBotQinZhenDotEnd, 255, -1)
    QinZhen = cv2.bitwise_and(image, image, mask=mask2)
    # ==============================================================================================================
    gray = cv2.cvtColor(QinZhen, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeftBotQinZhen = tuple(c[c[:, :, 0].argmin()][0])
    extRightBotQinZhen = tuple(c[c[:, :, 0].argmax()][0])
    extTopBotQinZhen = tuple(c[c[:, :, 1].argmin()][0])
    extBotBotQinZhen = tuple(c[c[:, :, 1].argmax()][0])
    rightTopDot_BotQinZhen = (extRightBotQinZhen[0], extTopBotQinZhen[1])
    leftTopDot_BotQinZhen = (extLeftBotQinZhen[0], extTopBotQinZhen[1])
    leftBottomDot_BotQinZhen = (extLeftBotQinZhen[0], extBotBotQinZhen[1])
    rightBottomDot_BotQinZhen = (extRightBotQinZhen[0], extBotBotQinZhen[1])
    topMiddlePoint_BotQinZhen = (
    leftTopDot_BotQinZhen[0] + ((rightTopDot_BotQinZhen[0] - leftTopDot_BotQinZhen[0]) // 2), rightTopDot_BotQinZhen[1])
    # cv2.rectangle(img_ori, leftTopDot_BotQinZhen, rightBottomDot_BotQinZhen, colorGreen, thickness)
    QinZhenCropped = image[leftTopDot_BotQinZhen[1]:rightBottomDot_BotQinZhen[1],
                     leftTopDot_BotQinZhen[0]:rightBottomDot_BotQinZhen[0]]
    # Draw to get the middle of Erhu ===============================================================================
    middleErhuStart = bottomQinZhenEnd
    middleErhuEnd = (middleErhuStart[0], middleErhuStart[1] + middleErhuLength)
    rightMidErhuDotStart = rightBotQinZhenDotEnd
    leftMidErhuDotStart = leftBotQinZhenDotEnd
    leftMidErhuDotEnd = (leftMidErhuDotStart[0], leftMidErhuDotStart[1] + middleErhuLength)
    rightMidErhuDotEnd = (rightMidErhuDotStart[0], rightMidErhuDotStart[1] + middleErhuLength)
    # cv2.rectangle(img_ori, leftMidErhuDotStart, rightMidErhuDotEnd, colorPurple, thickness)
    # === Remark for draw all segment of erhu ======================================================================
    mask3 = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask3, leftMidErhuDotStart, rightMidErhuDotEnd, 255, -1)
    mask_hand3 = cv2.bitwise_and(image, image, mask=mask3)
    # ==============================================================================================================
    # Draw to get the Qin Thong
    QinThongStart = middleErhuEnd
    QinThongEnd = centerPointBottom
    rightQinThongDotStart = rightMidErhuDotEnd
    leftQinThongDotStart = leftMidErhuDotEnd
    leftQinThongDotEnd = (leftQinThongDotStart[0], centerPointBottom[1])
    rightQinThongDotEnd = (rightQinThongDotStart[0], centerPointBottom[1])
    # cv2.rectangle(img_ori, leftQinThongDotStart, rightQinThongDotEnd, colorGreen, thickness)
    # === Remark for draw all segment of erhu ============================================
    mask4 = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask4, leftQinThongDotStart, rightQinThongDotEnd, 255, -1)
    mask_hand4 = cv2.bitwise_and(image, image, mask=mask4)
    # ====================================================================================
    # === Draw local boundary for Qin Thong ======================================================================
    QinThong = mask_hand4
    gray = cv2.cvtColor(QinThong[:, :], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeftQinThong = tuple(c[c[:, :, 0].argmin()][0])
    extRightQinThong = tuple(c[c[:, :, 0].argmax()][0])
    extTopQinThong = tuple(c[c[:, :, 1].argmin()][0])
    extBotQinThong = tuple(c[c[:, :, 1].argmax()][0])
    rightTopDot_QinThong = (extRightQinThong[0], extTopQinThong[1])
    leftTopDot_QinThong = (extLeftQinThong[0], extTopQinThong[1])
    leftBottomDot_QinThong = (extLeftQinThong[0], extBotQinThong[1])
    rightBottomDot_QinThong = (extRightQinThong[0], extBotQinThong[1])
    bottomMiddlePoint_QinThong = (
    leftBottomDot_QinThong[0] + ((rightBottomDot_QinThong[0] - leftBottomDot_QinThong[0]) // 3),
    rightBottomDot_QinThong[1])
    # cv2.rectangle(img_ori, leftTopDot_QinThong, rightBottomDot_QinThong, colorGreen, thickness)
    QinThongCropped = image[leftTopDot_QinThong[1]:rightBottomDot_QinThong[1],
                      leftTopDot_QinThong[0]:rightBottomDot_QinThong[0]]
    # cv2.line(img_ori, topMiddlePoint_Marker, bottomMiddlePoint_QinThong, colorRed, 2)
    erhu_line = [[topMiddlePoint_Marker[0], topMiddlePoint_Marker[1]],[bottomMiddlePoint_QinThong[0], bottomMiddlePoint_QinThong[1]]]
    # ==============================================================================================================
    # cv2.imshow("Image", img_ori)
    # cv2.waitKey(1)
    return img_ori, erhu_line, QinZhenCropped, QinThongCropped
    # return img_ori, erhu_line, QinZhen, mask_hand4


def calculate_area(p1, p2):
    p1x = p1[0]
    p1y = p1[1]
    p2x = p2[0]
    p2y = p2[1]
    if p1x > p2x:
        x_length = p1x - p2x
    else:
        x_length = p2x - p1x
    if p1y > p2y:
        y_length = p1y - p2y
    else:
        y_length = p2y - p1y
    value_of_area = x_length * y_length

    return value_of_area


def get_body_ratio(img):
    image = img.copy()
    gray = cv2.cvtColor(image[:, :], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeftImage = tuple(c[c[:, :, 0].argmin()][0])
    extRightImage = tuple(c[c[:, :, 0].argmax()][0])
    extTopImage = tuple(c[c[:, :, 1].argmin()][0])
    extBotImage = tuple(c[c[:, :, 1].argmax()][0])
    rightTopImage = (extRightImage[0], extTopImage[1])
    leftTopImage = (extLeftImage[0], extTopImage[1])
    leftBottomImage = (extLeftImage[0], extBotImage[1])
    rightBottomImage = (extRightImage[0], extBotImage[1])
    widthImage = image.shape[1]
    heightImage = image.shape[0]
    # print(widthImage, heightImage)
    value_body_areas = calculate_area(leftTopImage, rightBottomImage)
    value_img_areas = heightImage * widthImage
    # print(value_img_areas)
    percent_areas = round((value_body_areas / value_img_areas) * 100)
    # print(percent_areas)
    cv2.rectangle(image, leftTopImage, rightBottomImage, (0, 255, 0), 2)
    cv2.putText(image, 'L=' + str(value_body_areas) + ', Percent = ' + str(percent_areas), leftTopImage,
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    return image


def get_var_multiplier(w, h):
    # print('res:',w,h)
    if w <= 640 and h <= 480:
        var_multiplier = 0.5
        # print('640x480')
    elif (w > 640 and w <= 1280) and (h > 480 and h <= 720):
        var_multiplier = 1
        # print('1280x720')
    elif (w > 1280 and w <= 1920) and (h > 720 and h <= 1080):
        var_multiplier = 1.8
        # print('1920x1080')
    else:
        var_multiplier = 1
        # print('else')
    return var_multiplier


def cropped_by_pixel(img, p1X, p1Y, p2X, p2Y):
    image = img.copy()
    image_masker = np.zeros(image.shape[:2], np.uint8)
    image_rectangle_mask = cv2.rectangle(image_masker, (p1X, p1Y), (p2X, p2Y), 255, -1)
    image_bitwised = cv2.bitwise_and(image, image, mask=image_rectangle_mask)
    gray = cv2.cvtColor(image_bitwised[:, :], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeftPoint = tuple(c[c[:, :, 0].argmin()][0])
    extRightPoint = tuple(c[c[:, :, 0].argmax()][0])
    extTopPoint = tuple(c[c[:, :, 1].argmin()][0])
    extBotPoint = tuple(c[c[:, :, 1].argmax()][0])
    rightTopPoint = (extRightPoint[0], extTopPoint[1])
    leftTopPoint = (extLeftPoint[0], extTopPoint[1])
    leftBottomPoint = (extLeftPoint[0], extBotPoint[1])
    rightBottomPoint = (extRightPoint[0], extBotPoint[1])
    image_cropped = image[leftTopPoint[1]:rightBottomPoint[1], leftTopPoint[0]:rightBottomPoint[0]]
    return image_cropped

def get_cdf_hist(image_input):
    """
    Method to compute histogram and cumulative distribution function
    :param image_input: input image
    :return: cdf
    """
    hist, bins = np.histogram(image_input.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    return cdf_normalized

def clahe_enhancement(image, threshold, grid_size=(16, 16)):
    """
    Adaptive histogram equalization to enhance the input image
    :param image: input image
    :param threshold: clipping threshold
    :param grid_size: local neighbourhood
    :return: enhanced image
        """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=threshold, tileGridSize=grid_size)
    enhanced_image = clahe.apply(img)
    new_img =  cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)
    cdf = get_cdf_hist(enhanced_image)
    return new_img, cdf

def drop_backgroud(image):
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        results = holistic.process(image)
        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        canvas = np.zeros((image.shape[0], image.shape[1], 3), dtype="uint8")
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.pose_landmarks,
        #     mp_holistic.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles
        #         .get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.
        x_left_elbow1 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x * image_width) + 100
        y_left_elbow2 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y * image_height)
        x_right_thumb1 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].x * image_width) - 100
        x_right_elbow1 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x * image_width) - 100
        y_right_elbow2 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].y * image_height)

        if x_right_thumb1 < x_right_elbow1:
            right_most = x_right_thumb1
            left_most = x_left_elbow1
        else:
            right_most = x_right_elbow1
            left_most = x_left_elbow1

        new_img = image.copy()
        new_img[:, 0:right_most] = 255
        new_img[:, left_most: ] = 255
        # image brightness enhancer
        # enhanced_clahe, cdf_clahe = clahe_enhancement(new_img, 10)
        pil_img = Image.fromarray(new_img)
        enhancer = ImageEnhance.Contrast(pil_img)
        factor = 1  # increase contrast
        im_output = enhancer.enhance(factor)
        img_res = np.array(im_output)
        # im_output.save('more-contrast-image.png')
        # cv2.imshow('MediaPipe Holistic', enhanced_clahe)
        # cv2.waitKey(10000)
        # exit()
    return new_img


def body_segment(img, frame):
    image = frame.copy()
    img_ori = img.copy()
    image_ori = img.copy()
    colorGreen = (0, 255, 0)
    colorRed = (0, 0, 255)
    colorBlue = (255, 0, 0)
    colorYellow = (0, 255, 255)
    thickness = 0
    length_of_hand_boundaries = 50
    length_of_leftarm_boundaries_1 = [120, 80]
    length_of_leftarm_boundaries_2 = [30, 180]
    length_of_rightarm_boundaries_1 = [50, 80]
    length_of_rightarm_boundaries_2 = [80, 80]
    var_multiplier = get_var_multiplier(image.shape[1], image.shape[0])

    body_image = image
    gray = cv2.cvtColor(body_image[:, :], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeftBody = tuple(c[c[:, :, 0].argmin()][0])
    extRightBody = tuple(c[c[:, :, 0].argmax()][0])
    extTopBody = tuple(c[c[:, :, 1].argmin()][0])
    extBotBody = tuple(c[c[:, :, 1].argmax()][0])
    rightTopDot_Body = (extRightBody[0], extTopBody[1])
    leftTopDot_Body = (extLeftBody[0], extTopBody[1])
    leftBottomDot_Body = (extLeftBody[0], extBotBody[1])
    rightBottomDot_Body = (extRightBody[0], extBotBody[1])
    if rightTopDot_Body[0] > leftTopDot_Body[0]:
        middleTopBody = (leftTopDot_Body[0] + ((rightTopDot_Body[0] - leftTopDot_Body[0]) // 2), rightTopDot_Body[1])
        # middleBottomBody    = (leftBottomDot_Body[0] + ((rightBottomDot_Body[0] - leftBottomDot_Body[0])//2), rightBottomDot_Body[1])
        halfRectangleBodyCoordinate = (middleTopBody, rightBottomDot_Body)
    else:
        middleTopBody = (rightTopDot_Body[0] + ((leftTopDot_Body[0] - rightTopDot_Body[0]) // 2), leftTopDot_Body[1])
        # middleBottomBody = (rightBottomDot_Body[0] + ((leftBottomDot_Body[0] - rightBottomDot_Body[0]) // 2), leftBottomDot_Body[1])
        halfRectangleBodyCoordinate = (middleTopBody, leftBottomDot_Body)

    with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image_height, image_width, _ = image.shape
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Right hand thumb rectangle point
        x_R1 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].x * image_width + round(
            length_of_hand_boundaries * var_multiplier))
        x_R2 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].y * image_height - round(
            length_of_hand_boundaries * var_multiplier))
        x_R3 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].x * image_width - round(
            length_of_hand_boundaries * var_multiplier))
        x_R4 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].y * image_height + round(
            length_of_hand_boundaries * var_multiplier))
        if x_R1 > image_width: x_R1 = image_width
        if x_R2 < 0: x_R2 = 0
        if x_R3 < 0: x_R3 = 0
        if x_R4 > image_height: x_R4 = image_height
        # Left hand wrist/shoulder rectangle point
        x_L1 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width) + round(
            length_of_leftarm_boundaries_1[0] * var_multiplier)
        x_L2 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height) - round(
            length_of_leftarm_boundaries_1[1] * var_multiplier)
        x_L3 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width) - round(
            length_of_leftarm_boundaries_2[0] * var_multiplier)
        x_L4 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height) + round(
            length_of_leftarm_boundaries_2[1] * var_multiplier)
        # cv2.circle(img_ori,(x_L1, x_L2), 5, colorRed, 1)
        # cv2.circle(img_ori, (x_L3, x_L4), 5, colorBlue, 1)
        # Left hand thumb rectangle point
        x_L1a = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].x * image_width) + round(
            length_of_hand_boundaries * var_multiplier)
        x_L2a = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].y * image_height) - round(
            length_of_hand_boundaries * var_multiplier)
        x_L3a = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].x * image_width) - round(
            length_of_hand_boundaries * var_multiplier)
        x_L4a = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].y * image_height) + round(
            length_of_hand_boundaries * var_multiplier)
        # Face rectangle point
        x_L1f = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x * image_width) + 50
        x_L2f = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y * image_height) - 50
        x_L3f = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x * image_width) - 50
        x_L4f = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y * image_height) + 50
        if x_L1 > image_width: x_L1 = image_width
        if x_L2 < 0: x_L2 = 0
        if x_L3 < 0: x_L3 = 0
        if x_L4 > image_height: x_L4 = image_height
        if x_L1a > image_width: x_L1a = image_width
        if x_L2a < 0: x_L2a = 0
        if x_L3a < 0: x_L3a = 0
        if x_L4a > image_height: x_L4a = image_height
        x_L1_hip = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x * image_width)
        x_L2_hip = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y * image_height)
        x_R1_hip = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x * image_width)
        x_R2_hip = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y * image_height)
        hip_left_point = (x_L1_hip, x_L2_hip)
        hip_right_point = (x_R1_hip, x_R2_hip)

        x_L1_shoulders = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width)
        x_L2_shoulders = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height)
        x_R1_shoulders = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width)
        x_R2_shoulders = round(
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_height)

        x_L1_eye = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x * image_width)
        x_L2_eye = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y * image_height)
        x_R1_eye = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x * image_width)
        x_R2_eye = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y * image_height)

        x_L1_mouth = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].x * image_width)
        x_L2_mouth = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].y * image_height)
        x_R1_mouth = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].x * image_width)
        x_R2_mouth = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].y * image_height)

        x_L1_ear = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x * image_width)
        x_L2_ear = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y * image_height)
        x_R1_ear = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x * image_width)
        x_R2_ear = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y * image_height)

        x_L1_knees = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].x * image_width)
        x_L2_knees = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].y * image_height)
        x_R1_knees = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].x * image_width)
        x_R2_knees = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].y * image_height)

        x_l1_rightHandArm = round(
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width) + round(
            length_of_rightarm_boundaries_1[0] * var_multiplier)
        x_L2_rightHandArm = round(
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_height) - round(
            length_of_rightarm_boundaries_1[1] * var_multiplier)
        x_R1_rightHandArm = round(
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x * image_width) - round(
            length_of_rightarm_boundaries_2[0] * var_multiplier)
        x_R2_rightHandArm = round(
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y * image_height) + round(
            length_of_rightarm_boundaries_2[1] * var_multiplier)

        if x_R1_shoulders > x_L1_shoulders:
            middle_shoulder_point = (((x_R1_shoulders - x_L1_shoulders) // 2) + x_L1_shoulders, x_L2_shoulders)
        else:
            middle_shoulder_point = (((x_L1_shoulders - x_R1_shoulders) // 2) + x_R1_shoulders, x_L2_shoulders)
        if x_R1_hip > x_L1_hip:
            middle_hip_point = (((x_R1_hip - x_L1_hip) // 2) + x_L1_hip, x_L2_hip)
        else:
            middle_hip_point = (((x_L1_hip - x_R1_hip) // 2) + x_R1_hip, x_L2_hip)

        if x_R1_eye > x_L1_eye:
            middle_eye_point = (((x_R1_eye - x_L1_eye) // 2) + x_L1_eye, x_L2_eye)
        else:
            middle_eye_point = (((x_L1_eye - x_R1_eye) // 2) + x_R1_eye, x_L2_eye)
        if x_R1_mouth > x_L1_mouth:
            middle_mouth_point = (((x_R1_mouth - x_L1_mouth) // 2) + x_L1_mouth, x_L2_mouth)
        else:
            middle_mouth_point = (((x_L1_mouth - x_R1_mouth) // 2) + x_R1_mouth, x_L2_mouth)

        # left_arm           = cv2.rectangle(img_ori, (x_L1, x_L2), (x_L3, x_L4),  colorBlue, thickness)
        left_arm_coordinate = [(x_L1, x_L2), (x_L3, x_L4)]
        left_arm_cropped = cropped_by_pixel(image, x_L1, x_L2, x_L3, x_L4)
        # === Get right arm with original image shape ======================================================================
        mask_left_arm = np.zeros(image.shape[:2], np.uint8)
        cv2.rectangle(mask_left_arm, (x_L1, x_L2), (x_L3, x_L4), 255, -1)
        left_arm_ori_shape = cv2.bitwise_and(image, image, mask=mask_left_arm)

        # left_hand           = cv2.rectangle(left_arm, (x_R1, x_R2), (x_R3, x_R4), colorGreen, thickness)
        left_hand_cropped = cropped_by_pixel(image, x_R1, x_R2, x_R3, x_R4)
        # === Get right hand with original image shape ======================================================================
        mask_left_hand = np.zeros(image.shape[:2], np.uint8)
        cv2.rectangle(mask_left_hand, (x_R1, x_R2), (x_R3, x_R4), 255, -1)
        left_hand_ori_shape = cv2.bitwise_and(image, image, mask=mask_left_hand)

        # right_hand          = cv2.rectangle(img_ori, (x_L1a, x_L2a), (x_L3a, x_L4a), colorRed, thickness)
        right_hand_cropped = cropped_by_pixel(image, x_L1a, x_L2a, x_L3a, x_L4a)
        right_hand_coordinate = [(x_L1a, x_L2a), (x_L3a, x_L4a)]
        # === Get right hand with original image shape ======================================================================
        mask_right_hand = np.zeros(image.shape[:2], np.uint8)
        cv2.rectangle(mask_right_hand, (x_L1a, x_L2a), (x_L3a, x_L4a), 255, -1)
        right_hand_ori_shape = cv2.bitwise_and(image, image, mask=mask_right_hand)

        # rightHandArm        = cv2.rectangle(right_hand, (x_l1_rightHandArm, x_L2_rightHandArm), (x_R1_rightHandArm, x_R2_rightHandArm), colorBlue, thickness)
        right_arm_cropped = cropped_by_pixel(image, x_l1_rightHandArm, x_L2_rightHandArm, x_R1_rightHandArm,
                                             x_R2_rightHandArm)
        right_arm_coordinate = [(x_l1_rightHandArm, x_L2_rightHandArm), (x_R1_rightHandArm, x_R2_rightHandArm)]
        mask_right_arm = np.zeros(image.shape[:2], np.uint8)
        cv2.rectangle(mask_right_arm, (x_l1_rightHandArm, x_L2_rightHandArm), (x_R1_rightHandArm, x_R2_rightHandArm),
                      255, -1)
        right_arm_ori_shape = cv2.bitwise_and(image, image, mask=mask_right_arm)

        # left_hip        = cv2.circle(rightHandArm, (x_L1_hip, x_L2_hip), 10, colorRed, -1)
        # right_hip       = cv2.circle(left_hip, (x_R1_hip, x_R2_hip), 10, colorRed, -1)
        # left_shoulders  = cv2.circle(right_hip, (x_L1_shoulders, x_L2_shoulders), 10, colorRed, -1)
        # right_shoulders = cv2.circle(left_shoulders, (x_R1_shoulders, x_R2_shoulders), 10, colorRed, -1)
        # middle_shoulder = cv2.circle(right_shoulders, middle_shoulder_point, 1, colorRed, -1)
        # middle_hip      = cv2.circle(middle_shoulder, middle_hip_point, 1, colorRed, -1)
        # face_rectangle  = cv2.rectangle(middle_hip, (x_L1f, x_L2f), (x_L3f, x_L4f), colorRed, thickness)
        # draw_body_line  = cv2.line(face_rectangle, middle_shoulder_point, middle_hip_point, colorRed, 2)
        # draw_face_line  = cv2.line(draw_body_line, middle_eye_point, middle_mouth_point, colorRed, 2)
        # draw_shoulder_line      = cv2.line(draw_face_line, (x_L1_shoulders, x_L2_shoulders), (x_R1_shoulders, x_R2_shoulders), colorRed, 2)
        # draw_ear_face_line      = cv2.line(draw_shoulder_line, (x_L1_ear,x_L2_ear), (x_R1_ear, x_R2_ear), colorRed, thickness)
        # draw_left_knees_circle  = cv2.circle(draw_ear_face_line, (x_L1_knees, x_L2_knees), 10, colorRed, -1)
        # draw_right_knees_circle = cv2.circle(draw_left_knees_circle, (x_R1_knees, x_R2_knees), 10, colorRed, -1)
        degrees_body = get_angle(middle_shoulder_point, middle_hip_point)
        degrees_shoulder = get_angle((x_R1_shoulders, x_R2_shoulders), (x_L1_shoulders, x_L2_shoulders))
        degrees_ear_face = get_angle((x_R1_ear, x_R2_ear), (x_L1_ear, x_L2_ear))
        head_rectangle_coordinate = [(x_L1f, x_L2f), (x_L3f, x_L4f)]
        body_rectangle_coordinate = [(x_L1_shoulders, x_L2_shoulders), (x_R1_knees, x_R2_knees)]
        knee_shoulder_distance = x_R1_shoulders - x_R1_knees

        # halfBodyRectangle = cv2.rectangle(draw_right_knees_circle, halfRectangleBodyCoordinate[0], halfRectangleBodyCoordinate[1], colorBlue, thickness)
        half_body_cropped = cropped_by_pixel(image, halfRectangleBodyCoordinate[0][0],
                                             halfRectangleBodyCoordinate[0][1], halfRectangleBodyCoordinate[1][0],
                                             halfRectangleBodyCoordinate[1][1])
        # === Get half body with original image shape ======================================================================
        mask_half_body = np.zeros(image.shape[:2], np.uint8)
        cv2.rectangle(mask_half_body, halfRectangleBodyCoordinate[0], halfRectangleBodyCoordinate[1], 255, -1)
        half_body_ori_shape = cv2.bitwise_and(image, image, mask=mask_half_body)
        # ret_image = halfBodyRectangle
        ret_image = img_ori
    return ret_image, right_hand_cropped, right_arm_cropped, left_hand_cropped, left_arm_cropped, half_body_cropped, \
           left_arm_ori_shape, right_arm_ori_shape, left_hand_ori_shape, right_hand_ori_shape, \
           left_arm_coordinate, right_arm_coordinate, right_hand_coordinate, \
           head_rectangle_coordinate, body_rectangle_coordinate, \
           knee_shoulder_distance, degrees_ear_face, degrees_body, degrees_shoulder, hip_left_point,hip_right_point
    # return ret_image, right_hand_ori_shape, right_arm_ori_shape, left_hand_ori_shape, left_arm_ori_shape, half_body_ori_shape


def bow_segment(img_ori, frame):
    colorGreen = (0, 255, 0)
    colorRed = (0, 0, 255)
    colorBlue = (255, 0, 0)
    colorYellow = (0, 255, 255)
    thickness = 2
    image = frame.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # threshold the image, then perform a series of erosions dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # determine the most extreme points along the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    rightTopDot = (extRight[0], extTop[1])
    leftTopDot = (extLeft[0], extTop[1])
    leftBottomDot = (extLeft[0], extBot[1])
    rightBottomDot = (extRight[0], extBot[1])
    bow_cropped = image[leftTopDot[1]:rightBottomDot[1], leftTopDot[0]:rightBottomDot[0]]
    BowLength = leftTopDot[0] - rightTopDot[0]
    topBowLength = int((BowLength * 5) / 100)
    middleBowLength = int((BowLength * 90) / 100)
    bottomBowLength = int((BowLength * 5) / 100)
    # === Get start point of Bow Lines =========================================================================
    leftTopDot_TopBow = leftTopDot
    rightTopDot_TopBow = (leftTopDot[0] - topBowLength, leftTopDot[1])
    leftBotDot_TopBow = leftBottomDot
    rightBotDot_TopBow = (leftBottomDot[0] - topBowLength, leftBottomDot[1])
    mask1 = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask1, leftTopDot_TopBow, rightBotDot_TopBow, 255, -1)
    mask_bow1 = cv2.bitwise_and(image, image, mask=mask1)
    bow_1 = mask_bow1.copy()
    gray = cv2.cvtColor(bow_1, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # cv2.drawContours(image, [c], -1, (0, 255, 255), 1)
    extLeftBow1 = tuple(c[c[:, :, 0].argmin()][0])
    extRightBow1 = tuple(c[c[:, :, 0].argmax()][0])
    extTopBow1 = tuple(c[c[:, :, 1].argmin()][0])
    extBotBow1 = tuple(c[c[:, :, 1].argmax()][0])
    rightTopBow1 = (extRightBow1[0], extTopBow1[1])
    leftTopBow1 = (extLeftBow1[0], extTopBow1[1])
    leftBotBow1 = (extLeftBow1[0], extBotBow1[1])
    rightBotBow1 = (extRightBow1[0], extBotBow1[1])
    middlePoint_Bow1 = (((rightBotBow1[0] - leftTopBow1[0]) // 2) + leftTopBow1[0],
                        (((rightBotBow1[1] - leftTopBow1[1]) // 2) + leftTopBow1[1]))
    # === End of Get start point of Bow Lines ==================================================================
    # === Get start point of Bow Lines =========================================================================
    leftTopDot_MidBow = rightTopDot_TopBow
    rightTopDot_MidBow = (leftTopDot_MidBow[0] - middleBowLength, leftTopDot_MidBow[1])
    leftBotDot_MidBow = rightBotDot_TopBow
    rightBotDot_MidBow = (leftBotDot_MidBow[0] - middleBowLength, leftBotDot_MidBow[1])
    mask2 = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask2, leftTopDot_MidBow, rightBotDot_MidBow, 255, -1)
    mask_bow2 = cv2.bitwise_and(image, image, mask=mask2)
    bow_2 = mask_bow2.copy()
    gray = cv2.cvtColor(bow_2, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeftBow2 = tuple(c[c[:, :, 0].argmin()][0])
    extRightBow2 = tuple(c[c[:, :, 0].argmax()][0])
    extTopBow2 = tuple(c[c[:, :, 1].argmin()][0])
    extBotBow2 = tuple(c[c[:, :, 1].argmax()][0])
    rightTopBow2 = (extRightBow2[0], extTopBow2[1])
    lefTopBow2 = (extLeftBow2[0], extTopBow2[1])
    leftBotBow2 = (extLeftBow2[0], extBotBow2[1])
    rightBotBow2 = (extRightBow2[0], extBotBow2[1])
    # === End of Get start point of Bow Lines ==================================================================
    # === Get start point of Bow Lines =========================================================================
    leftTopDot_BotBow = rightTopDot_MidBow
    rightTopDot_BotBow = (leftTopDot_BotBow[0] - bottomBowLength, leftTopDot_BotBow[1])
    leftBotDot_BotBow = rightBotDot_MidBow
    rightBotDot_BotBow = (leftBotDot_BotBow[0] - bottomBowLength, leftBotDot_BotBow[1])
    mask3 = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask3, leftTopDot_BotBow, rightBotDot_BotBow, 255, -1)
    mask_bow2 = cv2.bitwise_and(image, image, mask=mask3)
    bow_2 = mask_bow2.copy()
    gray = cv2.cvtColor(bow_2, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeftBow3 = tuple(c[c[:, :, 0].argmin()][0])
    extRightBow3 = tuple(c[c[:, :, 0].argmax()][0])
    extTopBow3 = tuple(c[c[:, :, 1].argmin()][0])
    extBotBow3 = tuple(c[c[:, :, 1].argmax()][0])
    rightTopBow3 = (extRightBow3[0], extTopBow3[1])
    leftTopBow3 = (extLeftBow3[0], extTopBow3[1])
    leftBotBow3 = (extLeftBow3[0], extBotBow3[1])
    rightBotBow3 = (extRightBow3[0], extBotBow3[1])
    middlePoint_Bow3 = (((rightBotBow3[0] - leftTopBow3[0]) // 2) + leftTopBow3[0],
                        (((rightBotBow3[1] - leftTopBow3[1]) // 2) + leftTopBow3[1]))
    # === End of Get start point of Bow Lines ==================================================================

    if middlePoint_Bow1[1] < middlePoint_Bow3[1]:
        # cv2.line(img_ori, leftTopBow1, rightBotBow3, colorRed, thickness)
        bow_line = [[leftTopBow1[0], leftTopBow1[1]], [rightBotBow3[0], rightBotBow3[1]]]
        bow_angle = get_angle((leftTopBow1[0], leftTopBow1[1]), (rightBotBow3[0], rightBotBow3[1]))
    else:
        # cv2.line(img_ori, leftBotBow1, rightTopBow3, colorRed, thickness)
        bow_line = [[leftBotBow1[0], leftBotBow1[1]], [rightTopBow3[0], rightTopBow3[1]]]
        bow_angle = get_angle((leftBotBow1[0], leftBotBow1[1]), (rightTopBow3[0], rightTopBow3[1]))

    return img_ori, bow_line, bow_cropped, bow_angle


def getImages(folder):
    data = []
    for base, dirs, files in os.walk(folder):
        if len(dirs) == 0:
            data.append([base, files])
            print('base:', base)
            print('files:', files)
    return data

def new_erhu_segment(frame):
    img = frame.copy()
    grayIMG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tmp = np.argwhere(grayIMG > 0)
    topY = np.amin(tmp[:, 0])
    bottomY = np.amax(tmp[:, 0])
    tmp = np.argwhere(grayIMG[topY] > 0)
    topX = np.average(tmp)
    topLocation = topX, topY
    tmp = np.argwhere(grayIMG[bottomY] > 0)
    bottomX = np.average(tmp)
    bottomLocation = bottomX, bottomY

    while len(coordTop) >= smoothingLeng:
        coordTop.pop(0)
        coordBottom.pop(0)

    coordTop.append(topLocation)
    coordBottom.append(bottomLocation)
    topLocation = np.average(coordTop, axis=0)
    bottomLocation = np.average(coordBottom, axis=0)

    img = cv2.line(img, (int(topLocation[0]), int(topLocation[1])), (int(bottomLocation[0]), int(bottomLocation[1])),
                   (0, 255, 0), 2)
    erhu_line_coord = [(int(topLocation[0]), int(topLocation[1])), (int(bottomLocation[0]), int(bottomLocation[1]))]

    return img, erhu_line_coord


def new_bow_segment(frame):
    img = frame.copy()
    grayIMG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    tmp = np.argwhere(grayIMG > 0)
    leftX = np.amin(tmp[:, 1])
    rightX = np.amax(tmp[:, 1])

    tmp = np.argwhere(grayIMG[:, leftX] > 0)
    leftY = np.average(tmp)
    leftLocation = leftX, leftY

    tmp = np.argwhere(grayIMG[:, rightX] > 0)
    rightY = np.average(tmp)
    rightLocation = rightX, rightY

    while len(coordLeft) >= smoothingLeng:
        coordLeft.pop(0)
        coordRight.pop(0)

    coordLeft.append(leftLocation)
    coordRight.append(rightLocation)
    leftLocation = np.average(coordLeft, axis=0)
    rightLocation = np.average(coordRight, axis=0)

    img = cv2.line(img, (int(leftLocation[0]), int(leftLocation[1])), (int(rightLocation[0]), int(rightLocation[1])),
                   (0, 255, 0), 2)
    bow_line = [(int(leftLocation[0]), int(leftLocation[1])), (int(rightLocation[0]), int(rightLocation[1]))]

    return img, bow_line


def main_predict(video_input, isFlip = True):
    start_now = datetime.now()
    start_time = start_now.strftime("%H:%M:%S")
    curr_date = str(start_now.day) + '-' + str(start_now.month) + '-' + str(start_now.year)
    curr_time = str(start_now.strftime('%H%M%S'))

    # model_left_hand = load_model(os.path.join(thisfolder, 'model/dataset_ver5/final/models_leftHand/model-007.h5'))
    model_left_arm = load_model(os.path.join(thisfolder, 'model/PredictionModel/model_left_arm/model-009.h5'))
    model_right_hand = load_model(os.path.join(thisfolder, 'model/PredictionModel/model_right_hand/model-018.h5'))
    model_right_arm = load_model(os.path.join(thisfolder, 'model/PredictionModel/model_right_arm/model-015.h5'))
    # model_left = load_model('/home/minelab/dev/erhu-project/classifier/model/PredictionModel/model_left/model-008.h5')
    # model_right = load_model('/home/minelab/dev/erhu-project/classifier/model/PredictionModel/model_right/model-014.h5')
    # model_left.summary()
    # model_right.summary()
    # model_left_hand.summary()
    model_left_arm.summary()
    model_right_hand.summary()
    model_right_arm.summary()
    # print('TEST')
    # exit()
    result_folder = os.path.join(os.path.abspath(__file__ + "/../../"), "predict")
    if os.path.exists(result_folder) == False:
        os.mkdir(result_folder)

    # dataset_folder  = 'video/5th_dataset_test'
    # dataset         = getImages(dataset_folder)
    # video_path      = 'video/5th_datasets_10 Apr 2022/0409 Separate each group/E23.mp4'
    video_path = video_input
    filename = video_path.split("/")[-1]
    filename = filename[:-5]
    # print(video_path)
    # path = os.path.join(os.path.abspath(__file__ + "/../../"), "upload", video_path)

    videoInput = cv2.VideoCapture(video_path)
    LABELS = ["body", "bow", "erhu"]
    segment_image = custom_segmentation()
    segment_image.inferConfig(num_classes=3, class_names=LABELS)
    segment_image.load_model(
        os.path.join(thisfolder, "model/SegmentationModel/4_5_dataset_13032022/mask_rcnn_model.081-0.129956.h5"))
    properties = getVideoProperties(videoInput)
    # videoSegmentation = {}
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # vid_fps = properties['FPS']
    vid_fps = 30.0
    vid_width = properties['Width']
    vid_height = properties['Height']
    videoOut_1 = cv2.VideoWriter(os.path.join(result_folder, filename + ".mp4"), fourcc,
                                 vid_fps, (vid_width, vid_height))
    print(os.path.join(result_folder, filename + "_output_" + curr_time + ".mp4"))
    # videoOut_qinthong = cv2.VideoWriter(
    #     "result/ErhuErrorClassification_v1_17032022/allsegment_" + prefix + "_qinthong.mp4", fourcc, vid_fps,
    #     (vid_width, vid_height))
    # videoOut_qinzhen = cv2.VideoWriter(
    #     "result/ErhuErrorClassification_v1_17032022/allsegment_" + prefix + "_qinzhen.mp4", fourcc, vid_fps,
    #     (vid_width, vid_height))
    # videoOut_lefthand = cv2.VideoWriter(
    #     "result/ErhuErrorClassification_v1_17032022/allsegment_" + prefix + "_lefthand.mp4", fourcc, vid_fps,
    #     (vid_width, vid_height))
    # videoOut_righthand = cv2.VideoWriter(
    #     "result/ErhuErrorClassification_v1_17032022/allsegment_" + prefix + "_righthand.mp4", fourcc, vid_fps,
    #     (vid_width, vid_height))
    # videoOut_leftArm  = cv2.VideoWriter("result/Erhu_5th_PredictResult_17032022_v1/"+filename+"_leftArm.mp4", fourcc, vid_fps, (vid_width, vid_height))
    # videoOut_rightArm = cv2.VideoWriter("result/Erhu_5th_PredictResult_17032022_v1/"+filename+"_rightArm.mp4", fourcc, vid_fps, (vid_width, vid_height))
    # videoOut_bow = cv2.VideoWriter(
    #     "result/ErhuErrorClassification_v1_19022022/allsegment_" + prefix + "_bow.mp4", fourcc, vid_fps,
    #     (vid_width, vid_height))
    # videoOut_mask       = cv2.VideoWriter("result/ErhuErrorClassification_v1_18022022/allsegment_1fps_" + prefix + "_mask.mp4", fourcc, 1, (vid_width, vid_height))
    frame_number = 0
    frame_bow = 0
    frame_body = 0
    frame_erhu = 0
    skipped_frame = 0
    selected_frame = 10
    max_frame = 9999
    img_height, img_width = 112, 112
    limit_sample = 30
    x_test_bow = []
    x_test_leftArm = []
    x_test_leftHand = []
    x_test_rightArm = []
    x_test_rightHand = []
    x_test_qinThong = []
    x_test_qinZhen = []
    x_test_video_bow_resized = []
    x_test_video_leftArm_resized = []
    x_test_video_leftHand_resized = []
    x_test_video_rightArm_resized = []
    x_test_video_rightHand_resized = []
    x_test_video_qinThong_resized = []
    x_test_video_qinZhen_resized = []
    x_test_video_bow = []
    x_test_video_leftArm = []
    x_test_video_leftHand = []
    x_test_video_rightArm = []
    x_test_video_rightHand = []
    x_test_video_qinThong = []
    x_test_video_qinZhen = []
    x_test_video_original = []
    x_test_right_hand_point = []
    x_test_right_arm_point = []
    x_test_left_arm_point = []
    x_test_all_body_point = []
    x_test_erhu_line_point = []
    x_test_bow_line_point = []
    x_test_bow_point = []
    output_array = []
    y_test = []
    frame_start = 10
    frame_stop = 10
    limit_counter = 0
    limit_counter_bow = 0
    limit_counter_body = 0
    limit_counter_erhu = 0
    # left_arm_point  = []
    # right_arm_point = []
    # right_hand_point = []
    first_frame = True
    while videoInput.isOpened():
        # if (frame_number > frame_start) :
        if limit_counter < limit_sample:
            # if (limit_counter_body < limit_sample) or (limit_counter_erhu < limit_sample) or (limit_counter_bow < limit_sample):
            success, frame = videoInput.read()
            if not success:
                break
            # if isFlip == True:
            # frame = cv2.flip(frame, 1)
            # cv2.imshow('flip', cv2.flip(frame.copy(), 1))
            # cv2.waitKey(10)
            try:
                # droped_img = drop_backgroud(frame.copy())
                seg_mask, seg_output = segment_image.segmentFrame(frame.copy())
                segLeng = len(seg_mask['scores'])
                # print(seg_mask['scores'])
                # print('Number of Segment:', segLeng)
                if segLeng >= 3:
                    # img_body = body_dilation(frame.copy())
                    image = frame.copy()
                    # print("Frame ", str(frame_number), " processed!")
                    for i in range(segLeng):
                        mask = frame.copy()
                        id = seg_mask['class_ids'][i]
                        label = LABELS[int(id) - 1]
                        if mask.shape[0] == seg_mask['masks'].shape[0] and mask.shape[1] == seg_mask['masks'].shape[1]:
                            mask[seg_mask['masks'][:, :, i] == False] = (0, 0, 0)
                            # videoSegmentation[label].write(mask)
                            if label == 'erhu':
                                # cv2.imshow('erhu', mask)
                                # cv2.waitKey(1)
                                # counter_erhu +=1
                                img, erhu_line = new_erhu_segment(mask)
                                # image, erhu_line, QinZhenPart, QinThongPart = erhu_segment(image, mask)
                                limit_counter_erhu += 1
                                if len(x_test_erhu_line_point) < limit_sample:
                                    x_test_erhu_line_point.append(erhu_line)
                                # if len(x_test_video_qinThong_resized) < limit_sample:
                                #     img_resized_qinThong = cv2.resize(QinThongPart, (img_height, img_width))
                                #     x_test_video_qinThong_resized.append(img_resized_qinThong)
                                #     x_test_video_qinThong.append(mask)
                                # if len(x_test_video_qinZhen_resized) < limit_sample:
                                #     img_resized_qinZhen = cv2.resize(QinZhenPart, (img_height, img_width))
                                #     x_test_video_qinZhen_resized.append(img_resized_qinZhen)
                                #     x_test_video_qinZhen.append(mask)
                            elif label == 'body':
                                # counter_body += 1
                                # cv2.imshow('flip', image)
                                # cv2.waitKey(1000)
                                image, rightHandPart, rightArmPart, leftHandPart, leftArmPart, \
                                halfBodyPart, leftArmPartOri, rightArmPartOri, leftHandPartOri, \
                                rightHandPartOri, left_arm_point, right_arm_point, right_hand_point, head_coordinate, body_coordinate, \
                                knees_shoulder_distance, degree_ear_face, degree_body, degree_shoulder, hip_left, hip_right = body_segment(
                                    image, mask)
                                limit_counter_body += 1
                                # cv2.imshow('Mask', image)
                                # cv2.imshow('leftHand', leftHandPart)
                                # cv2.imshow('leftArm', leftArmPart)
                                # cv2.imshow('RightArm', rightArmPart)
                                # cv2.imshow('RightHand', rightHandPart)
                                # cv2.waitKey(1)
                                if len(x_test_video_leftHand_resized) < limit_sample:
                                    img_resized_leftHand = cv2.resize(leftHandPart, (img_height, img_width))
                                    x_test_video_leftHand_resized.append(img_resized_leftHand)
                                    x_test_video_leftHand.append(leftHandPartOri)
                                if len(x_test_video_rightHand_resized) < limit_sample:
                                    img_resized_rightHand = cv2.resize(rightHandPart, (img_height, img_width))
                                    x_test_video_rightHand_resized.append(img_resized_rightHand)
                                    x_test_video_rightHand.append(rightHandPartOri)
                                    x_test_right_hand_point.append(right_hand_point)
                                if len(x_test_video_leftArm_resized) < limit_sample:
                                    img_resized_leftArm = cv2.resize(leftArmPart, (img_height, img_width))
                                    x_test_video_leftArm_resized.append(img_resized_leftArm)
                                    x_test_video_leftArm.append(leftArmPartOri)
                                    x_test_left_arm_point.append(left_arm_point)
                                if len(x_test_video_rightArm_resized) < limit_sample:
                                    img_resized_rightArm = cv2.resize(rightArmPart, (img_height, img_width))
                                    x_test_video_rightArm_resized.append(img_resized_rightArm)
                                    x_test_video_rightArm.append(rightArmPartOri)
                                    x_test_right_arm_point.append(right_arm_point)
                                if len(x_test_all_body_point) < limit_sample:
                                    x_test_all_body_point.append(
                                        [head_coordinate, body_coordinate, knees_shoulder_distance, degree_ear_face,
                                         degree_body, degree_shoulder, hip_left, hip_right])
                            elif label == 'bow':
                                # cv2.imshow('bow', mask)
                                # cv2.waitKey(1)
                                # counter_bow += 1
                                img, bow_line = new_bow_segment(mask)
                                # image, bow_line, bow_cropped, bow_angle = bow_segment(image, mask)
                                limit_counter_bow += 1
                                if len(x_test_video_bow_resized) < limit_sample:
                                    # img_resized_bow = cv2.resize(bow_cropped, (img_height, img_width))
                                    # x_test_video_bow_resized.append(img_resized_bow)
                                    x_test_bow_line_point.append(bow_line)
                                    # x_test_video_bow.append(mask)
                                    # x_test_bow_point.append(bow_angle)
                    # is_orthogonal = check_orthogonal(erhu_line[0][0], erhu_line[1][1], erhu_line[1][0], erhu_line[1][1], bow_line[0][0], bow_line[0][1], bow_line[1][0], bow_line[1][1])
                    x_test_video_original.append(image)
                    frame_number += 1
                    limit_counter += 1
            except:
                print("Something is wrong...")
            # # else:
            # #     skipped_frame += 1
        else:

            limit_counter = 0
            limit_counter_bow = 0
            limit_counter_body = 0
            limit_counter_erhu = 0
            # continue
            # frame_number +=1
            # print('Original Image Count:', len(x_test_video_original))
            # print('Image Bow Count:', len(x_test_video_bow_resized))
            # print('Image LeftArm Count:', len(x_test_video_leftArm_resized))
            # print('Image LeftHand Count:', len(x_test_video_leftHand_resized))
            # print('Image RightArm Count:', len(x_test_video_rightArm_resized))
            # print('Image RightHand Count:', len(x_test_video_rightHand_resized))
            # print('Image QinThong Count:', len(x_test_video_qinThong_resized))
            # print('Image QinZhen Count:', len(x_test_video_qinZhen_resized))
            # x_test_bow.append(x_test_video_bow_resized)
            x_test_leftArm.append(x_test_video_leftArm_resized)
            x_test_leftHand.append(x_test_video_leftHand_resized)
            x_test_rightArm.append(x_test_video_rightArm_resized)
            x_test_rightHand.append(x_test_video_rightHand_resized)
            x_test_qinThong.append(x_test_video_qinThong_resized)
            x_test_qinZhen.append(x_test_video_qinZhen_resized)

            # test_x_bow          = np.array(x_test_bow)
            # print('bow shape:', test_x_bow.shape)
            # test_x_bow          = test_x_bow / 255.0

            test_x_leftArm = np.array(x_test_leftArm)
            # print('leftArm shape:', test_x_leftArm.shape)
            test_x_leftArm = test_x_leftArm / 255.0

            test_x_leftHand = np.array(x_test_leftHand)
            # print('leftHand shape:', test_x_leftHand.shape)
            test_x_leftHand = test_x_leftHand / 255.0

            test_x_rightArm = np.array(x_test_rightArm)
            # print('rightArm shape:', test_x_rightArm.shape)
            test_x_rightArm = test_x_rightArm / 255.0

            test_x_rightHand = np.array(x_test_rightHand)
            # print('rightHand shape:', test_x_rightHand.shape)
            test_x_rightHand = test_x_rightHand / 255.0

            # test_x_qinThong     = np.array(x_test_qinThong)
            # print('qinThong shape:', test_x_qinThong.shape)
            # test_x_qinThong     = test_x_qinThong / 255.0
            #
            # test_x_qinZhen      = np.array(x_test_qinZhen)
            # print('qinZhen shape:', test_x_qinZhen.shape)
            # test_x_qinZhen      = test_x_qinZhen / 255.0

            # prediction_leftArm  = model_left_arm.predict(test_x_leftArm)
            # prediction_leftHand = model_left_hand.predict(test_x_leftHand)
            # prediction_rightArm = model_right_arm.predict(test_x_rightArm)
            # prediction_rightHand = model_right_hand.predict(test_x_rightHand)
            prediction_leftArm = model_left_arm.predict(test_x_leftArm)
            # prediction_leftHand = model_left.predict(test_x_leftHand)
            prediction_rightArm = model_right_arm.predict(test_x_rightArm)
            prediction_rightHand = model_right_hand.predict(test_x_rightHand)

            print(prediction_rightArm)
            print(prediction_rightHand)
            print(prediction_leftArm)
            # print(prediction_leftHand)
            prediction_right_arm_max = np.argmax(prediction_rightArm[0])
            prediction_left_max = np.argmax(prediction_leftArm[0])
            prediction_right_hand_max = np.argmax(prediction_rightHand[0])
            rightArm_E31 = 0
            rightArm_E32 = 0
            rightArm_E33 = 0
            rightArm_E34 = 0
            rightArm_E35 = 0
            leftArm_E21 = 0
            leftArm_E22 = 0
            leftArm_E23 = 0

            if prediction_right_hand_max == 0:
                rightArm_E31 = round((prediction_rightHand[0][0]), 2)
            elif prediction_right_hand_max == 1:
                rightArm_E32 = round((prediction_rightHand[0][1]), 2)
            elif prediction_right_hand_max == 2:
                rightArm_E33 = round((prediction_rightHand[0][2]), 2)

            if prediction_right_arm_max == 0:
                rightArm_E34 = round((prediction_rightArm[0][0]), 2)
            else:
                rightArm_E35 = round((prediction_rightArm[0][1]), 2)

            if prediction_left_max == 0:
                leftArm_E21 = round((prediction_leftArm[0][0]), 2)
            elif prediction_left_max == 1:
                leftArm_E22 = round((prediction_leftArm[0][1]), 2)
            else:
                leftArm_E23 = round((prediction_leftArm[0][2]), 2)

            # rightArm_E31 = round(prediction_rightHand[0][3],2)
            # rightArm_E32 = round(prediction_rightHand[0][4],2)
            # rightArm_E33 = round(prediction_rightHand[0][5],2)
            # rightArm_E34 = round(prediction_rightArm[0][6],2)
            # rightArm_E35 = round(prediction_rightArm[0][7],2)
            # rightArm_E31 = round(np.argmax(prediction_rightHand), 2)
            # rightArm_E32 = round(np.argmax(prediction_rightHand), 2)
            # rightArm_E33 = round(np.argmax(prediction_rightHand), 2)
            # rightArm_E34 = round(np.argmax(prediction_rightArm), 2)
            # rightArm_E35 = round(np.argmax(prediction_rightArm), 2)
            rightArm_E31_ClassName = 'E31-拇指握弓位置錯誤'
            rightArm_E32_ClassName = 'E32-食指握弓桿位置錯誤'
            rightArm_E33_ClassName = 'E33-中指或無名指觸弓位置錯誤'
            rightArm_E34_ClassName = 'E34-右手腕持弓太向內翻'
            rightArm_E35_ClassName = 'E35-右手腕持弓太向外翻'
            rightArm_Normal_ClassName = 'N-右手正常持弓 '
            # leftArm_E21 = round(np.argmax(prediction_leftArm),2)
            # leftArm_E22 = round(np.argmax(prediction_leftArm),2)
            # leftArm_E23 = round(np.argmax(prediction_leftHand),2)
            print('leftArm_E21:', leftArm_E21)
            print('leftArm_E22:', leftArm_E22)
            print('leftArm_E23:', leftArm_E23)
            print('rightArm_E31:', rightArm_E31)
            print('rightArm_E32:', rightArm_E32)
            print('rightArm_E33:', rightArm_E33)
            print('rightArm_E34:', rightArm_E34)
            print('rightArm_E35:', rightArm_E35)
            leftArm_Normal_ClassName = 'N-左手正常'
            leftArm_E21_ClassName = 'E21-左手肘過高'
            leftArm_E22_ClassName = 'E22-左手肘太低'
            leftArm_E23_ClassName = 'E23-手腕手肘放輕鬆連成一直線'
            E11_classname = 'E11-頭要擺正'
            E14_classname = 'E14-坐姿要正'
            E15_classname = 'E15-兩膝與肩同寬'
            E13_classname = 'E13-右肩太高'
            E12_classname = 'E12-右肩太高'
            E41_classname = 'E41-琴桿左傾斜 - 弓毛與琴弦必須垂直'
            E42_classname = 'E42-琴桿右傾斜 - 弓毛與琴弦必須垂直'
            E43_classname = 'E43-運弓軌跡必須保持一直線'
            err_face = 0
            err_body = 0
            err_orthogonal = 0
            err_bow = 0
            err_erhu_left = 0
            err_erhu_right = 0
            err_right_shoulder = 0
            err_left_shoulder = 0
            err_knees_shoulder = 0
            is_face_err = False
            is_body_err = False
            is_left_shoulder_err = False
            is_right_shoulder_err = False
            is_bow_err = False
            is_knees_shoulder = False
            is_erhu_left = False
            is_erhu_right = False
            is_not_orthogonal = False
            for (all_body_point, bow_line, erhu_line) in zip(x_test_all_body_point, x_test_bow_line_point, x_test_erhu_line_point):
                head_coordinate = all_body_point[0]
                body_coordinate = all_body_point[1]
                knees_shoulder_distance = all_body_point[2]
                degree_ear_face = all_body_point[3]
                degree_body = all_body_point[4]
                degree_shoulder = all_body_point[5]
                hip_left_point = all_body_point[6]
                hip_right_point = all_body_point[7]

                if degree_ear_face >= K_var or degree_ear_face < (0 - K_var):
                    err_face += 1
                if degree_body > (90 + K_var) or degree_body < (90 - K_var):
                    err_body += 1
                if degree_shoulder > ((X_var / 90) * 100):
                    err_right_shoulder += 1
                elif degree_shoulder < (0 - ((X_var / 90) * 100)):
                    err_left_shoulder += 1
                if abs(knees_shoulder_distance) > K_var:
                    err_knees_shoulder += 1

                L1_angle = get_angle(erhu_line[0], erhu_line[1])
                L2_angle = get_angle(bow_line[0], bow_line[1])
                orthogonal_angle = abs(abs(L1_angle) - abs(L2_angle))
                if (orthogonal_angle <= 90 - N_var or orthogonal_angle >= 90 + N_var):
                    err_orthogonal += 1
                    if L1_angle > (90 + N_var):
                        err_erhu_left += 1
                    else:
                        err_erhu_right += 1
                elif (L2_angle > (0 + N_var) or L2_angle < (0 - N_var)):
                    err_bow += 1

            if err_face > limit_sample // 2:
                is_face_err = True
            if err_bow > limit_sample // 2:
                is_bow_err = True
            if err_orthogonal > limit_sample // 2:
                is_not_orthogonal = True
            if err_erhu_left > limit_sample // 2:
                is_erhu_left = True
            if err_erhu_right > limit_sample // 2:
                is_erhu_right = True
            if err_body > limit_sample // 2:
                is_body_err = True
            if err_left_shoulder > limit_sample // 2:
                is_left_shoulder_err = True
            if err_right_shoulder > limit_sample // 2:
                is_right_shoulder_err = True
            if err_knees_shoulder > limit_sample // 2:
                is_knees_shoulder = True

            for img_original, ori_right_hand_point, ori_right_arm_point, ori_left_arm_point, all_body_point, bow_line, erhu_line in \
                    zip(x_test_video_original, x_test_right_hand_point, x_test_right_arm_point, x_test_left_arm_point,
                        x_test_all_body_point, x_test_bow_line_point, x_test_erhu_line_point):
                warning_mess = []
                img_pil = Image.fromarray(img_original)
                draw_res_bow = ImageDraw.Draw(img_pil)
                # print('Rectangle:', right_hand_point, right_arm_point, left_arm_point)
                ori_right_hand_rectangle_shape = [ori_right_hand_point[0], ori_right_hand_point[1]]
                ori_right_arm_rectangle_shape = [ori_right_arm_point[0], ori_right_arm_point[1]]
                # print('Shape:', ori_right_hand_rectangle_shape, ori_right_arm_rectangle_shape)
                # x_test_all_body_point.append([head_coordinate, body_coordinate, knees_shoulder_distance, degree_ear_face, degree_body, degree_shoulder])
                head_coordinate = all_body_point[0]
                body_coordinate = all_body_point[1]
                knees_shoulder_distance = all_body_point[2]
                degree_ear_face = all_body_point[3]
                degree_body = all_body_point[4]
                degree_shoulder = all_body_point[5]
                hip_left_point = all_body_point[6]
                hip_right_point = all_body_point[7]

                # cv2.rectangle(img_chinese, (10, 10), (200, 200), (255, 255, 255))
                # draw_res_bow.rectangle([(5, 5), (500, 350)], outline=None, fill="#ffffff")
                cv2.rectangle(img_chinese, (10, 10), (100, 100), (255, 255, 255))
                # draw_res_bow.rectangle([(5, 5), (300, 150)], outline=None, fill="#ffffff")

                # draw_res_bow.text((10, 20), "===== Traditional Algorithm =====", font=result_font, fill=(b, g, r, a))

                # E11 : Head position not normal ===================================================================== 0
                # if degree_ear_face >= K_var or degree_ear_face < (0-K_var):
                if is_face_err == True:
                    # print('Face')
                    # draw_res_bow.text((10, 50), E11_classname + ':' + str(degree_ear_face), font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["E11", str(degree_ear_face), 'Head Position', str(degree_ear_face),
                                         'E11-Head position not normal (to L or R)'])
                    draw_res_bow.rectangle(head_coordinate, outline="blue", fill=None, width=4)
                else:
                    # print('Else Face')
                    # draw_res_bow.text((10, 50), 'Head Position : Normal', font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["Head_Normal", "Normal", 'Head Position', str(0), 'Normal'])

                # E14 : Error Body Position ========================================================================== 1
                # if degree_body > (90+K_var) or degree_body < (90-K_var):
                if is_body_err == True:
                    # print('Body')
                    # draw_res_bow.text((10, 80), E14_classname + ':' + str(degree_body), font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["E14", str(degree_body), 'Body Position', str(degree_body),
                                         'E14-Need to seat straight (to L or R)'])
                    draw_res_bow.rectangle(body_coordinate, outline="blue", fill=None, width=4)
                else:
                    # print('Else Body')
                    # draw_res_bow.text((10, 80), 'Body Position : Normal', font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["Body_Normal", "Normal", 'Body Position', str(0), 'Normal'])

                # E12 - E13 Left Shoulder too High / Right Shoulder too High ========================================= 2
                # if degree_shoulder > ((X_var/90)*100):
                if is_right_shoulder_err == True:
                    # print('Shoulder')
                    # draw_res_bow.text((10, 140), E13_classname + ':' + str(degree_shoulder), font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["E13", str(degree_shoulder), 'Shoulder Position', str(degree_shoulder),
                                         'E13-Right shoulder too hight '])
                    draw_res_bow.rectangle(body_coordinate, outline="blue", fill=None, width=4)
                # elif degree_shoulder < (0-((X_var/90)*100)):
                elif is_left_shoulder_err == True:
                    # print('Else Shoulder')
                    # draw_res_bow.text((10, 140), E12_classname + ':' + str(degree_shoulder), font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["E12", str(degree_shoulder), 'Shoulder Position', str(degree_shoulder),
                                         'E12-Left shoulder too hight'])
                    draw_res_bow.rectangle(body_coordinate, outline="blue", fill=None, width=4)
                else:
                    # print('Else Shoulder')
                    # draw_res_bow.text((10, 140), 'Shoulders Position : Normal', font=result_font, fill=(b, g, r, a))
                    warning_mess.append(['Shoulder_Normal', 'Normal', 'Shoulders Position', str(0), 'Normal'])

                # E31 Error Right Arm ================================================================================ 3
                # draw_res_bow.text((10, 170), "===== Right Arm =====", font=result_font, fill=(b, g, r, a))
                if rightArm_E31 >= P_var:
                    # print('Right Arm')
                    # draw_res_bow.text((10, 200), rightArm_E31_ClassName + ':' + str(rightArm_E31), font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["E31", str(rightArm_E31), 'Right Hand Arm Position', str(rightArm_E31),
                                         'E31-Wrong RH thumb position'])
                    draw_res_bow.rectangle(ori_right_hand_rectangle_shape, outline="blue", fill=None, width=4)
                elif rightArm_E32 >= P_var:
                    # print('Else Right Arm')
                    # draw_res_bow.text((10, 200), rightArm_E32_ClassName + ':' + str(rightArm_E32), font=result_font,fill=(b, g, r, a))
                    warning_mess.append(["E32", str(rightArm_E32), 'Right Hand Arm Position', str(rightArm_E32),
                                         'E32-Wrong RH index finger position'])
                    draw_res_bow.rectangle(ori_right_hand_rectangle_shape, outline="blue", fill=None, width=4)
                elif rightArm_E33 >= P_var:
                    # print('Else Right Arm')
                    # draw_res_bow.text((10, 200), rightArm_E33_ClassName + ':' + str(rightArm_E33), font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["E33", str(rightArm_E33), 'Right Hand Arm Position', str(rightArm_E33),
                                         'E33-Wrong RH middle or ring finger position'])
                    draw_res_bow.rectangle(ori_right_hand_rectangle_shape, outline="blue", fill=None, width=4)
                elif rightArm_E34 >= P_var:
                    # print('Else Right Arm')
                    # draw_res_bow.text((10, 200), rightArm_E34_ClassName + ':' + str(rightArm_E34), font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["E34", str(rightArm_E34), 'Right Hand Arm Position', str(rightArm_E34),
                                         'E34-Right wrist position too inward'])
                    draw_res_bow.rectangle(ori_right_arm_rectangle_shape, outline="blue", fill=None, width=4)
                elif rightArm_E35 >= P_var:
                    # print('Else Right Arm')
                    # draw_res_bow.text((10, 200), rightArm_E35_ClassName + ':' + str(rightArm_E35), font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["E35", str(rightArm_E35), 'Right Hand Arm Position', str(rightArm_E35),
                                         'E35-Right wrist position too outward'])
                    draw_res_bow.rectangle(ori_right_arm_rectangle_shape, outline="blue", fill=None, width=4)
                else:
                    # print('Else Right Arm')
                    # draw_res_bow.text((10, 200), rightArm_Normal_ClassName, font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["RightArm_Normal", 'Normal', "Right Arm Position", str(1.0), 'Normal'])

                # E21 Error Left Arm ================================================================================= 4
                # draw_res_bow.text((10, 230), "===== Left Arm =====", font=result_font, fill=(b, g, r, a))
                ori_left_arm_rectangle_shape = [ori_left_arm_point[0], ori_left_arm_point[1]]
                if leftArm_E21 >= P_var:
                    # print('Left Arm')
                    # draw_res_bow.text((10, 260), leftArm_E21_ClassName + ':' + str(leftArm_E21), font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["E21", str(leftArm_E21), 'Left Hand Arm Position', str(leftArm_E21),
                                         'E21-Left elbow too Hight'])
                    draw_res_bow.rectangle(ori_left_arm_rectangle_shape, outline='blue', fill=None, width=4)
                elif leftArm_E22 >= P_var:
                    # print('Else Left Arm')
                    # draw_res_bow.text((10, 260), leftArm_E22_ClassName + ':' + str(leftArm_E22), font=result_font, fill=(b, g, r, a))
                    warning_mess.append(
                        ["E22", str(leftArm_E22), 'Left Hand Arm Position', str(leftArm_E22), 'E22-Left elbow too Low'])
                    draw_res_bow.rectangle(ori_left_arm_rectangle_shape, outline='blue', fill=None, width=4)
                elif leftArm_E23 >= P_var:
                    # print('Else Left Arm')
                    # draw_res_bow.text((10, 260), leftArm_E23_ClassName + ':' + str(leftArm_E23), font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["E23", str(leftArm_E23), 'Left Hand Arm Position', str(leftArm_E23),
                                         'E23-Left elbow and wrist in a line'])
                    draw_res_bow.rectangle(ori_left_arm_rectangle_shape, outline='blue', fill=None, width=4)
                else:
                    # print('Else Left Arm')
                    # draw_res_bow.text((10, 260), leftArm_Normal_ClassName, font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["LeftArm_Normal", "Normal", 'Left Hand Arm Position', str(1.0), 'Normal'])

                # E41 Error Bow Erhu ================================================================================= 5
                # bow_line_shape = [(bow_line[0][0], bow_line[0][1]), (bow_line[1][0], bow_line[1][1])]
                # erhu_line_shape = [(erhu_line[0][0], erhu_line[0][1]), hip_left_point]
                erhu_line_shape = erhu_line
                bow_line_shape = bow_line
                # print('Erhu Bow Line Shape:', bow_line_shape, erhu_line_shape)
                # if is_orthogonal == False:
                if is_not_orthogonal == True:
                    L1_angle = get_angle(erhu_line[0], erhu_line[1])
                    L2_angle = get_angle(bow_line[0], bow_line[1])
                    if is_erhu_left == True:
                        # draw_res_bow.text((10, 320), E41_classname, font=result_font, fill=(b, g, r, a))
                        warning_mess.append(["E41", str(L1_angle), "Erhu Position", str(L1_angle),
                                             "E41-Pole tilt to left - Bow hair and string must be orthogonal"])
                        draw_res_bow.line(bow_line_shape, fill='blue', width=4)
                        draw_res_bow.line(erhu_line_shape, fill='blue', width=4)
                    else:
                        # draw_res_bow.text((10, 320), E42_classname, font=result_font, fill=(b, g, r, a))
                        warning_mess.append(["E42", str(L1_angle), "Erhu Position", str(L1_angle),
                                             "E42-Pole tilt to right - Bow hair and string must be orthogonal"])
                        draw_res_bow.line(bow_line_shape, fill='blue', width=4)
                        draw_res_bow.line(erhu_line_shape, fill='blue', width=4)
                elif is_bow_err == True:
                    # draw_res_bow.text((10, 320), E43_classname, font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["E43", str(L2_angle), "Bow Position", str(L2_angle),
                                         "E43-Trace of bow must be in straight line"])
                    draw_res_bow.line(bow_line_shape, fill='blue', width=4)
                    # draw_res_bow.line(erhu_line_shape, fill='blue', width=4)
                else:
                    # print('Else Orthogonal')
                    # draw_res_bow.text((10, 320), 'Bow and Erhu is Orthogonal', font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["Bow_normal", "Normal", "Bow Erhu Position", str(1.0), "Normal"])

                # E15 Error Knees Position =========================================================================== 6
                if is_knees_shoulder == True:
                    # draw_res_bow.text((10, 110), E15_classname + ':' + str(knees_shoulder_distance), font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["E15", str(knees_shoulder_distance), 'Knees Position', str(degree_body),
                                         'E15-Put knees in normal position'])
                    # draw_res_bow.rectangle((hip_left_point, hip_right_point), outline="blue", fill=None, width=4)
                else:
                    # draw_res_bow.text((10, 110), 'Knees Position : Normal', font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["Knees_Normal", "Normal", 'Knees Position', str(0), 'Normal'])
                resize_img = (256, 256)
                img_resized = np.array(img_pil.resize(resize_img))
                img_original = np.array(img_pil)
                # output_array.append([warning_mess, img_original])
                output_array.append([warning_mess])
                videoOut_1.write(img_original)
            #
            # for img_leftArm, left_arm_point in zip(x_test_video_leftArm, x_test_left_arm_point):
            #     img_pil = Image.fromarray(img_leftArm)
            #     draw_res_bow = ImageDraw.Draw(img_pil)
            #     left_arm_rectangle_shape = [left_arm_point[0], left_arm_point[1]]
            #     draw_res_bow.text((10, 20), "===== Left Arm =====", font=result_font, fill=(b, g, r, a))
            #     if leftArm_E21 >= P_var:
            #         draw_res_bow.text((10, 50), leftArm_E21_ClassName + ':' + str(leftArm_E21), font=result_font, fill=(b, g, r, a))
            #         draw_res_bow.rectangle(left_arm_rectangle_shape, outline='blue', fill=None, width=4)
            #     elif leftArm_E22 >= P_var:
            #         draw_res_bow.text((10, 50), leftArm_E22_ClassName + ':' + str(leftArm_E22), font=result_font, fill=(b, g, r, a))
            #         draw_res_bow.rectangle(left_arm_rectangle_shape, outline='blue', fill=None, width=4)
            #     elif leftArm_E23 >= P_var:
            #         draw_res_bow.text((10, 50), leftArm_E23_ClassName + ':' + str(leftArm_E23), font=result_font, fill=(b, g, r, a))
            #         draw_res_bow.rectangle(left_arm_rectangle_shape, outline='blue', fill=None, width=4)
            #     else:
            #         draw_res_bow.text((10, 50), leftArm_Normal_ClassName, font=result_font, fill=(b, g, r, a))
            #     img_leftArm = np.array(img_pil)
            #     videoOut_leftArm.write(img_leftArm)
            #
            # for img_rightArm, right_hand_point, right_arm_point in zip(x_test_video_rightArm, x_test_right_hand_point, x_test_right_arm_point):
            #     img_pil = Image.fromarray(img_rightArm)
            #     draw_res_bow = ImageDraw.Draw(img_pil)
            #     right_hand_rectangle_shape = [right_hand_point[0], right_hand_point[1]]
            #     right_arm_rectangle_shape = [right_arm_point[0], right_arm_point[1]]
            #     draw_res_bow.text((10, 20), "===== Right Arm =====", font=result_font, fill=(b, g, r, a))
            #     if rightArm_E31 >= P_var:
            #         draw_res_bow.text((10, 50), rightArm_E31_ClassName + ':' + str(rightArm_E31), font=result_font, fill=(b, g, r, a))
            #         draw_res_bow.rectangle(right_hand_rectangle_shape, outline="blue", fill=None, width=4)
            #     elif rightArm_E32 >= P_var:
            #         draw_res_bow.text((10, 50), rightArm_E32_ClassName + ':' + str(rightArm_E32), font=result_font, fill=(b, g, r, a))
            #         draw_res_bow.rectangle(right_hand_rectangle_shape, outline="blue", fill=None, width=4)
            #     elif rightArm_E33 >= P_var:
            #         draw_res_bow.text((10, 50), rightArm_E33_ClassName + ':' + str(rightArm_E33), font=result_font, fill=(b, g, r, a))
            #         draw_res_bow.rectangle(right_hand_rectangle_shape, outline="blue", fill=None, width=4)
            #     elif rightArm_E34 >= P_var:
            #         draw_res_bow.text((10, 50), rightArm_E34_ClassName + ':' + str(rightArm_E34), font=result_font, fill=(b, g, r, a))
            #         draw_res_bow.rectangle(right_arm_rectangle_shape, outline="blue", fill=None, width=4)
            #     elif rightArm_E35 >= P_var:
            #         draw_res_bow.text((10, 50), rightArm_E35_ClassName + ':' + str(rightArm_E35), font=result_font, fill=(b, g, r, a))
            #         draw_res_bow.rectangle(right_arm_rectangle_shape, outline="blue", fill=None, width=4)
            #     else:
            #         draw_res_bow.text((10, 50), rightArm_Normal_ClassName, font=result_font, fill=(b, g, r, a))
            #     img_rightArm = np.array(img_pil)
            #     videoOut_rightArm.write(img_rightArm)
            #
            # for bow_line, erhu_line in zip (x_test_bow_line_point, x_test_erhu_line_point):
            #     is_orthogonal = check_orthogonal(erhu_line[0][0], erhu_line[1][1], erhu_line[1][0], erhu_line[1][1], bow_line[0][0], bow_line[0][1], bow_line[1][0], bow_line[1][1])
            #     bow_line_shape = [(bow_line[0][0],bow_line[0][1]), (bow_line[1][0], bow_line[1][1])]
            #     erhu_line_shape = [(erhu_line[0][0], erhu_line[0][1]), (erhu_line[1][0], erhu_line[1][1])]
            #     if is_orthogonal == False:
            #         draw_res_bow.text((10, 230), E41_classname , font=result_font,fill=(b, g, r, a))
            #         # draw_res_bow.line(bow_line_shape, fill='blue', width=4)
            #         # draw_res_bow.line(erhu_line_shape, fill='blue', width=4)

            x_test_bow = []
            x_test_leftArm = []
            x_test_leftHand = []
            x_test_rightArm = []
            x_test_rightHand = []
            x_test_qinThong = []
            x_test_qinZhen = []
            x_test_video_bow_resized = []
            x_test_video_leftArm_resized = []
            x_test_video_leftHand_resized = []
            x_test_video_rightArm_resized = []
            x_test_video_rightHand_resized = []
            x_test_video_qinThong_resized = []
            x_test_video_qinZhen_resized = []
            x_test_video_bow = []
            x_test_video_leftArm = []
            x_test_video_leftHand = []
            x_test_video_rightArm = []
            x_test_video_rightHand = []
            x_test_video_qinThong = []
            x_test_video_qinZhen = []
            x_test_video_original = []
            x_test_right_hand_point = []
            x_test_right_arm_point = []
            x_test_left_arm_point = []
            x_test_all_body_point = []
            x_test_erhu_line_point = []
            x_test_bow_line_point = []
            x_test_bow_point = []
            # break
        # else:
        #     frame_number += 1
    # print(output_array)
    # np.savez_compressed(os.path.join(result_folder, filename + "_" + curr_time), output_array)
    np.savez_compressed(os.path.join(result_folder, filename), output_array)
    videoInput.release()
    tensorflow.keras.backend.clear_session()
    videoOut_1.release()
    # videoOut_qinthong.release()
    # videoOut_qinzhen.release()
    # videoOut_lefthand.release()
    # videoOut_righthand.release()
    # videoOut_leftArm.release()
    # videoOut_rightArm.release()
    # videoOut_bow.release()
    # videoOut_mask.release()
    # for label in videoSegmentation:
    #     videoSegmentation[label].release()
    end_now = datetime.now()
    end_time = end_now.strftime("%H:%M:%S")
    print("Start Time =", start_time)
    print("End Time =", end_time)

    return os.path.join(result_folder, filename + ".mp4")
