# Erhu Error Classification System ver. final
# Date created : 11 / 04 / 2022
# Current Progress :
# - Combined extract segmentation with prediction
# - Using newest 5th dataset from professor (10 April 2022)
import codecs
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
import warnings
import tensorflow
from pixellib.instance import custom_segmentation
from datetime import datetime
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image, ImageEnhance
from configparser import ConfigParser
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo
from mmdet.apis import inference_detector, init_detector
from keras_gcn import GraphConv
from keras_gcn import GraphMaxPool, GraphAveragePool

sys.path.append(os.path.abspath(os.path.join('..', 'upload')))

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)

tensorflow.config.experimental.enable_tensor_float_32_execution(
    False
)

configure = ConfigParser()
thisfolder = os.path.dirname(os.path.abspath(__file__))
configini = os.path.join(thisfolder, 'config.ini')
configure.read(configini)

det_model = init_detector(os.path.join(thisfolder, "mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"),
                          os.path.join(thisfolder, "model/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"),
                          device="cuda:0")
pose_model = init_pose_model(
    os.path.join(thisfolder, "mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py"),
    os.path.join(thisfolder, "model/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"),
    device="cuda:0")

dataset = pose_model.cfg.data['test']['type']
dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
if dataset_info is None:
    warnings.warn(
        'Please set `dataset_info` in the config. Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
        DeprecationWarning)
else:
    dataset_info = DatasetInfo(dataset_info)

for i in range(23, 92):
    dataset_info.pose_kpt_color[i] = [0, 0, 255]

return_heatmap = False
output_layer_names = None
thresHold = 0.7


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
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
bow_line = [[0, 0], [0, 0]]
erhu_line = [[0, 0], [0, 0]]

shoulders_left_point = []
shoulders_right_point = []
hip_left_point = []
hip_right_point = []



K_var = configure.getint('traditional', 'k')
K_LArm_high_var = configure.getint('traditional', 'h')
K_LArm_low_var = configure.getint('traditional', 'l')
X_var = configure.getint('traditional', 'x')
Y_var = configure.getint('traditional', 'y')
N_var = configure.getint('traditional', 'n')
M_var = configure.getint('traditional', 'm')
P_var = configure.getfloat('deeplearning', 'p')
S_var = configure.getfloat('traditional', 's') # Slope threshold
KN_var = configure.getfloat('traditional', 'kn') # Knees threshold

print("K_var:", K_var)
print("X_var:", X_var)
print("Y_var:", Y_var)
print("N_var:", N_var)
print("M_var:", M_var)
print("P_var:", P_var)
print("S_var:", S_var)
print("KN_var:", KN_var)

img_chinese = np.zeros((200, 400, 3), np.uint8)
b, g, r, a = 0, 255, 0, 0
chinese_font = './util/simsun.ttc'

# result_font = ImageFont.truetype(os.path.join(thisfolder, 'util/simsun.ttc'), 28)
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

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

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
    try:
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
    except:
        image_cropped = img.copy()
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
    try:
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
    except:
        ret_image, right_hand_cropped, right_arm_cropped, left_hand_cropped, left_arm_cropped, half_body_cropped, \
        left_arm_ori_shape, right_arm_ori_shape, left_hand_ori_shape, right_hand_ori_shape = img_ori, img_ori, img_ori, \
                                                                                             img_ori, img_ori, img_ori, \
                                                                                             img_ori, img_ori, img_ori, \
                                                                                             img_ori
        left_arm_coordinate, right_arm_coordinate, right_hand_coordinate, \
        head_rectangle_coordinate, body_rectangle_coordinate = [(0, 0), (0, 0)], [(0, 0), (0, 0)], [(0, 0), (0, 0)], \
                                                               [(0, 0), (0, 0)], [(0, 0), (0, 0)]
        knee_shoulder_distance, degrees_ear_face, degrees_body, degrees_shoulder = 0, 0, 0, 0
        hip_left_point, hip_right_point = (0, 0), (0, 0)
    return ret_image, right_hand_cropped, right_arm_cropped, left_hand_cropped, left_arm_cropped, half_body_cropped, \
           left_arm_ori_shape, right_arm_ori_shape, left_hand_ori_shape, right_hand_ori_shape, \
           left_arm_coordinate, right_arm_coordinate, right_hand_coordinate, \
           head_rectangle_coordinate, body_rectangle_coordinate, \
           knee_shoulder_distance, degrees_ear_face, degrees_body, degrees_shoulder, hip_left_point,hip_right_point
    # return ret_image, right_hand_ori_shape, right_arm_ori_shape, left_hand_ori_shape, left_arm_ori_shape, half_body_ori_shape

# def new_body_segment(frame):
#     image = frame.copy()
#     img_ori = frame.copy()
#     length_of_hand_boundaries = 50
#     length_of_leftarm_boundaries_1 = [120, 80]
#     length_of_leftarm_boundaries_2 = [30, 180]
#     length_of_rightarm_boundaries_1 = [50, 80]
#     length_of_rightarm_boundaries_2 = [80, 80]
#     var_multiplier = get_var_multiplier(image.shape[1], image.shape[0])
#     body_image = image
#     gray = cv2.cvtColor(body_image[:, :], cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
#     thresh = cv2.erode(thresh, None, iterations=2)
#     thresh = cv2.dilate(thresh, None, iterations=2)
#     cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     c = max(cnts, key=cv2.contourArea)
#     extLeftBody = tuple(c[c[:, :, 0].argmin()][0])
#     extRightBody = tuple(c[c[:, :, 0].argmax()][0])
#     extTopBody = tuple(c[c[:, :, 1].argmin()][0])
#     extBotBody = tuple(c[c[:, :, 1].argmax()][0])
#     rightTopDot_Body = (extRightBody[0], extTopBody[1])
#     leftTopDot_Body = (extLeftBody[0], extTopBody[1])
#     leftBottomDot_Body = (extLeftBody[0], extBotBody[1])
#     rightBottomDot_Body = (extRightBody[0], extBotBody[1])
#     if rightTopDot_Body[0] > leftTopDot_Body[0]:
#         middleTopBody = (leftTopDot_Body[0] + ((rightTopDot_Body[0] - leftTopDot_Body[0]) // 2), rightTopDot_Body[1])
#         halfRectangleBodyCoordinate = (middleTopBody, rightBottomDot_Body)
#     else:
#         middleTopBody = (rightTopDot_Body[0] + ((leftTopDot_Body[0] - rightTopDot_Body[0]) // 2), leftTopDot_Body[1])
#         halfRectangleBodyCoordinate = (middleTopBody, leftBottomDot_Body)
#     try:
#         with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
#             image.flags.writeable = False
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = holistic.process(image)
#             image_height, image_width, _ = image.shape
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#             # Right hand thumb rectangle point
#             x_R1 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].x * image_width + round(
#                 length_of_hand_boundaries * var_multiplier))
#             x_R2 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].y * image_height - round(
#                 length_of_hand_boundaries * var_multiplier))
#             x_R3 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].x * image_width - round(
#                 length_of_hand_boundaries * var_multiplier))
#             x_R4 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].y * image_height + round(
#                 length_of_hand_boundaries * var_multiplier))
#             if x_R1 > image_width: x_R1 = image_width
#             if x_R2 < 0: x_R2 = 0
#             if x_R3 < 0: x_R3 = 0
#             if x_R4 > image_height: x_R4 = image_height
#             # Left hand wrist/shoulder rectangle point
#             x_L1 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width) + round(
#                 length_of_leftarm_boundaries_1[0] * var_multiplier)
#             x_L2 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height) - round(
#                 length_of_leftarm_boundaries_1[1] * var_multiplier)
#             x_L3 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width) - round(
#                 length_of_leftarm_boundaries_2[0] * var_multiplier)
#             x_L4 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height) + round(
#                 length_of_leftarm_boundaries_2[1] * var_multiplier)
#             # Left hand thumb rectangle point
#             x_L1a = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].x * image_width) + round(
#                 length_of_hand_boundaries * var_multiplier)
#             x_L2a = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].y * image_height) - round(
#                 length_of_hand_boundaries * var_multiplier)
#             x_L3a = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].x * image_width) - round(
#                 length_of_hand_boundaries * var_multiplier)
#             x_L4a = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].y * image_height) + round(
#                 length_of_hand_boundaries * var_multiplier)
#             # Face rectangle point
#             x_L1f = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x * image_width) + 50
#             x_L2f = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y * image_height) - 50
#             x_L3f = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x * image_width) - 50
#             x_L4f = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y * image_height) + 50
#             if x_L1 > image_width: x_L1 = image_width
#             if x_L2 < 0: x_L2 = 0
#             if x_L3 < 0: x_L3 = 0
#             if x_L4 > image_height: x_L4 = image_height
#             if x_L1a > image_width: x_L1a = image_width
#             if x_L2a < 0: x_L2a = 0
#             if x_L3a < 0: x_L3a = 0
#             if x_L4a > image_height: x_L4a = image_height
#             x_L1_hip = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x * image_width)
#             x_L2_hip = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y * image_height)
#             x_R1_hip = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x * image_width)
#             x_R2_hip = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y * image_height)
#             hip_left_point = (x_L1_hip, x_L2_hip)
#             hip_right_point = (x_R1_hip, x_R2_hip)
#
#             x_L1_shoulders = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width)
#             x_L2_shoulders = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height)
#             x_R1_shoulders = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width)
#             x_R2_shoulders = round(
#                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_height)
#
#             x_L1_eye = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x * image_width)
#             x_L2_eye = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y * image_height)
#             x_R1_eye = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x * image_width)
#             x_R2_eye = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y * image_height)
#
#             x_L1_mouth = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].x * image_width)
#             x_L2_mouth = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_LEFT].y * image_height)
#             x_R1_mouth = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].x * image_width)
#             x_R2_mouth = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.MOUTH_RIGHT].y * image_height)
#
#             x_L1_ear = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x * image_width)
#             x_L2_ear = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y * image_height)
#             x_R1_ear = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x * image_width)
#             x_R2_ear = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y * image_height)
#
#             x_L1_knees = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].x * image_width)
#             x_L2_knees = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].y * image_height)
#             x_R1_knees = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].x * image_width)
#             x_R2_knees = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].y * image_height)
#
#             x_l1_rightHandArm = round(
#                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width) + round(
#                 length_of_rightarm_boundaries_1[0] * var_multiplier)
#             x_L2_rightHandArm = round(
#                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_height) - round(
#                 length_of_rightarm_boundaries_1[1] * var_multiplier)
#             x_R1_rightHandArm = round(
#                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x * image_width) - round(
#                 length_of_rightarm_boundaries_2[0] * var_multiplier)
#             x_R2_rightHandArm = round(
#                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y * image_height) + round(
#                 length_of_rightarm_boundaries_2[1] * var_multiplier)
#
#             if x_R1_shoulders > x_L1_shoulders:
#                 middle_shoulder_point = (((x_R1_shoulders - x_L1_shoulders) // 2) + x_L1_shoulders, x_L2_shoulders)
#             else:
#                 middle_shoulder_point = (((x_L1_shoulders - x_R1_shoulders) // 2) + x_R1_shoulders, x_L2_shoulders)
#             if x_R1_hip > x_L1_hip:
#                 middle_hip_point = (((x_R1_hip - x_L1_hip) // 2) + x_L1_hip, x_L2_hip)
#             else:
#                 middle_hip_point = (((x_L1_hip - x_R1_hip) // 2) + x_R1_hip, x_L2_hip)
#
#             if x_R1_eye > x_L1_eye:
#                 middle_eye_point = (((x_R1_eye - x_L1_eye) // 2) + x_L1_eye, x_L2_eye)
#             else:
#                 middle_eye_point = (((x_L1_eye - x_R1_eye) // 2) + x_R1_eye, x_L2_eye)
#             if x_R1_mouth > x_L1_mouth:
#                 middle_mouth_point = (((x_R1_mouth - x_L1_mouth) // 2) + x_L1_mouth, x_L2_mouth)
#             else:
#                 middle_mouth_point = (((x_L1_mouth - x_R1_mouth) // 2) + x_R1_mouth, x_L2_mouth)
#             left_arm_coordinate = [(x_L1, x_L2), (x_L3, x_L4)]
#             left_arm_cropped = cropped_by_pixel(image, x_L1, x_L2, x_L3, x_L4)
#             # === Get right arm with original image shape ======================================================================
#             mask_left_arm = np.zeros(image.shape[:2], np.uint8)
#             cv2.rectangle(mask_left_arm, (x_L1, x_L2), (x_L3, x_L4), 255, -1)
#             left_arm_ori_shape = cv2.bitwise_and(image, image, mask=mask_left_arm)
#             left_hand_cropped = cropped_by_pixel(image, x_R1, x_R2, x_R3, x_R4)
#             # === Get right hand with original image shape ======================================================================
#             mask_left_hand = np.zeros(image.shape[:2], np.uint8)
#             cv2.rectangle(mask_left_hand, (x_R1, x_R2), (x_R3, x_R4), 255, -1)
#             left_hand_ori_shape = cv2.bitwise_and(image, image, mask=mask_left_hand)
#             right_hand_cropped = cropped_by_pixel(image, x_L1a, x_L2a, x_L3a, x_L4a)
#             right_hand_coordinate = [(x_L1a, x_L2a), (x_L3a, x_L4a)]
#             # === Get right hand with original image shape ======================================================================
#             mask_right_hand = np.zeros(image.shape[:2], np.uint8)
#             cv2.rectangle(mask_right_hand, (x_L1a, x_L2a), (x_L3a, x_L4a), 255, -1)
#             right_hand_ori_shape = cv2.bitwise_and(image, image, mask=mask_right_hand)
#             right_arm_cropped = cropped_by_pixel(image, x_l1_rightHandArm, x_L2_rightHandArm, x_R1_rightHandArm, x_R2_rightHandArm)
#             right_arm_coordinate = [(x_l1_rightHandArm, x_L2_rightHandArm), (x_R1_rightHandArm, x_R2_rightHandArm)]
#             mask_right_arm = np.zeros(image.shape[:2], np.uint8)
#             cv2.rectangle(mask_right_arm, (x_l1_rightHandArm, x_L2_rightHandArm), (x_R1_rightHandArm, x_R2_rightHandArm), 255, -1)
#             right_arm_ori_shape = cv2.bitwise_and(image, image, mask=mask_right_arm)
#             degrees_body = get_angle(middle_shoulder_point, middle_hip_point)
#             degrees_shoulder = get_angle((x_R1_shoulders, x_R2_shoulders), (x_L1_shoulders, x_L2_shoulders))
#             degrees_ear_face = get_angle((x_R1_ear, x_R2_ear), (x_L1_ear, x_L2_ear))
#             head_rectangle_coordinate = [(x_L1f, x_L2f), (x_L3f, x_L4f)]
#             body_rectangle_coordinate = [(x_L1_shoulders, x_L2_shoulders), (x_R1_knees, x_R2_knees)]
#             knee_shoulder_distance = x_R1_shoulders - x_R1_knees
#             half_body_cropped = cropped_by_pixel(image, halfRectangleBodyCoordinate[0][0],
#                                                  halfRectangleBodyCoordinate[0][1], halfRectangleBodyCoordinate[1][0],
#                                                  halfRectangleBodyCoordinate[1][1])
#             # === Get half body with original image shape ======================================================================
#             mask_half_body = np.zeros(image.shape[:2], np.uint8)
#             cv2.rectangle(mask_half_body, halfRectangleBodyCoordinate[0], halfRectangleBodyCoordinate[1], 255, -1)
#             half_body_ori_shape = cv2.bitwise_and(image, image, mask=mask_half_body)
#             # ret_image = halfBodyRectangle
#             ret_image = img_ori
#     except:
#         ret_image, right_hand_cropped, right_arm_cropped, left_hand_cropped, left_arm_cropped, half_body_cropped, \
#         left_arm_ori_shape, right_arm_ori_shape, left_hand_ori_shape, right_hand_ori_shape = img_ori, img_ori, img_ori, \
#                                                                                              img_ori, img_ori, img_ori, \
#                                                                                              img_ori, img_ori, img_ori, \
#                                                                                              img_ori
#         left_arm_coordinate, right_arm_coordinate, right_hand_coordinate, \
#         head_rectangle_coordinate, body_rectangle_coordinate = [(0, 0), (0, 0)], [(0, 0), (0, 0)], [(0, 0), (0, 0)], \
#                                                                [(0, 0), (0, 0)], [(0, 0), (0, 0)]
#         knee_shoulder_distance, degrees_ear_face, degrees_body, degrees_shoulder = 0, 0, 0, 0
#         hip_left_point, hip_right_point = (0, 0), (0, 0)
#     return ret_image, right_hand_cropped, right_arm_cropped, left_hand_cropped, left_arm_cropped, half_body_cropped, \
#            left_arm_ori_shape, right_arm_ori_shape, left_hand_ori_shape, right_hand_ori_shape, \
#            left_arm_coordinate, right_arm_coordinate, right_hand_coordinate, \
#            head_rectangle_coordinate, body_rectangle_coordinate, \
#            knee_shoulder_distance, degrees_ear_face, degrees_body, degrees_shoulder, hip_left_point,hip_right_point

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
    try:
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
    except:
        img = frame.copy()
        erhu_line_coord = [(0, 0),(0, 0)]

    return img, erhu_line_coord


def new_bow_segment(frame):
    try:
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
    except:
        img = frame.copy()
        bow_line = [(0, 0), (0, 0)]
    return img, bow_line


# def body_landmark_segment(frame):
#     image = frame.copy()
#     img_ori = frame.copy()
#     try:
#         with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
#             image.flags.writeable = False
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = holistic.process(image)
#             image_height, image_width, _ = image.shape
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#             # mp_drawing.draw_landmarks(
#             #     img_ori,
#             #     results.left_hand_landmarks,
#             #     mp_holistic.HAND_CONNECTIONS,
#             #     landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
#             # Get Wrist Right Hand
#             x_WRRH = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x * image_width)
#             y_WRRH = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y * image_height)
#             rh_wrist = [x_WRRH, y_WRRH]
#
#             # Get Wrist Left Hand
#             x_WRLH = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x * image_width)
#             y_WRLH = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y * image_height)
#             lh_wrist = [x_WRLH, y_WRLH]
#
#             # Get Index Finger Right Hand
#             x_IDXFGRRH = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].x * image_width)
#             y_IDXFGRRH = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_INDEX].y * image_height)
#             rh_index_finger = [x_IDXFGRRH, y_IDXFGRRH]
#
#
#             try:
#                 # Get Middle Finger Left Hand Hand Landmark
#                 x_MIDFGRLH = round(results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
#                 y_MIDFGRLH = round(results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
#                 lh_middle_finger = [x_MIDFGRLH, y_MIDFGRLH]
#                 # cv2.circle(img_ori, lh_middle_finger, 5 (255, 0, 0), -1)
#             except:
#                 # Get Index Finger Left Hand Pose Landmark
#                 x_IDXFGRLH = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_INDEX].x * image_width)
#                 y_IDXFGRLH = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_INDEX].y * image_height)
#                 lh_index_finger = [x_IDXFGRLH, y_IDXFGRLH]
#                 x_PKYFGRLH = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_PINKY].x * image_width)
#                 y_PKYFGRLH = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_PINKY].y * image_height)
#                 lh_pinky_finger = [x_PKYFGRLH, y_PKYFGRLH]
#                 # Get Left Middle Finger Wrist Line
#                 lh_middle_finger = [lh_index_finger[0] - (abs(lh_index_finger[1] - lh_pinky_finger[1])), (abs(lh_index_finger[1] - lh_pinky_finger[1]) // 4) + lh_index_finger[1]]
#                 # cv2.circle(img_ori, lh_middle_finger, 5, (0, 0, 255), -1)
#
#             # Get Shoulder Right Hand
#             x_R_shoulder = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width)
#             y_R_shoulder = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_height)
#             rh_shoulder = [x_R_shoulder, y_R_shoulder]
#
#             # Get Shoulder Left Hand
#             x_L_shoulder = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width)
#             y_L_shoulder = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height)
#             lh_shoulder = [x_L_shoulder, y_L_shoulder]
#             diff_shoulder = abs(x_L_shoulder-x_R_shoulder)
#
#             # Get Elbow Right Hand
#             x_R_elbow = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x * image_width)
#             y_R_elbow = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y * image_height)
#             rh_elbow = [x_R_elbow, y_R_elbow]
#
#             # Get Elbow Left Hand
#             x_L_elbow = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x * image_width)
#             y_L_elbow = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y * image_height)
#             lh_elbow = [x_L_elbow, y_L_elbow]
#
#             # Get Right Knees
#             x_R_knees = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].x * image_width)
#             y_R_knees = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE].y * image_height)
#             rh_knees = [x_R_knees, y_R_knees]
#
#             # Get Left Knees
#             x_L_knees = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].x * image_width)
#             y_L_knees = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].y * image_height)
#             lh_knees = [x_L_knees, y_L_knees]
#             diff_knees = abs(x_L_knees-x_R_knees)
#
#             # Get Left Ear
#             x_L_ear = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x * image_width)
#             y_L_ear = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y * image_height)
#             lh_ear = [x_L_ear, y_L_ear]
#
#             # Get Right Ear
#             x_R_ear = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x * image_width)
#             y_R_ear = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y * image_height)
#             rh_ear = [x_R_ear, y_R_ear]
#
#             # Get Hip point
#             x_R_hip = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].x * image_width)
#             y_R_hip = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP].y * image_height)
#             x_L_hip = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x * image_width)
#             y_L_hip = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y * image_height)
#             rh_hip = [x_R_hip, y_R_hip]
#             lh_hip = [x_L_hip, y_L_hip]
#
#             # Get Head Box
#             x_L_eye = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x * image_width) + (abs(x_R_shoulder - x_L_shoulder) // 4)
#             y_L_eye = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y * image_height) - (abs(x_R_shoulder - x_L_shoulder) // 4)
#             x_R_eye = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].x * image_width) - (abs(x_R_shoulder - x_L_shoulder) // 4)
#             y_R_eye = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EYE].y * image_height) + (abs(x_R_shoulder - x_L_shoulder) // 4)
#             head_rectangle_coordinate = [(x_L_eye, y_L_eye), (x_R_eye, y_R_eye)]
#
#             # Get Middle Body
#             if x_R_shoulder > x_L_shoulder:
#                 middle_shoulder_point = ((abs(x_R_shoulder - x_L_shoulder) // 2) + x_L_shoulder, y_L_shoulder)
#             else:
#                 middle_shoulder_point = ((abs(x_L_shoulder - x_R_shoulder) // 2) + x_R_shoulder, y_R_shoulder)
#             if x_R_hip > x_L_hip:
#                 middle_hip_point = (abs((x_R_hip - x_L_hip) // 2) + x_L_hip, y_L_hip)
#             else:
#                 middle_hip_point = (abs((x_L_hip - x_R_hip) // 2) + x_R_hip, y_R_hip)
#
#
#
#             # Get Finger Wrist Slope Left Hand Middle Finger
#             lh_finger_wrist_degree = get_angle(lh_middle_finger, lh_wrist)
#
#             # Get Wrist Elbow Slope Left Hand Elbow
#             lh_wrist_elbow_degree = get_angle(lh_wrist, lh_elbow)
#
#             lh_slope_value = abs(lh_wrist_elbow_degree - lh_finger_wrist_degree)
#             lh_finger_wrist_elbow_coor = [lh_middle_finger, lh_wrist, lh_elbow]
#
#             print('lh_slop_middle_finger_wrist:', lh_finger_wrist_degree)
#             print('lh_slop_wrist_elbow:', lh_wrist_elbow_degree)
#             # cv2.line(img_ori, lh_middle_finger, lh_wrist, (0, 255, 0), 3)
#             # cv2.line(img_ori, lh_wrist, lh_elbow, (0, 255, 0), 3)
#
#
#             # Get Left Shoulder Elbow Line
#             degree_lh_shoulder_elbow = get_angle(lh_shoulder, lh_elbow)
#             # cv2.line(img_ori, lh_shoulder, lh_elbow, (0, 0, 255), 3)
#
#             # Get Body Box
#             body_rectangle_coordinate = [(x_L_shoulder, y_L_shoulder), (x_R_hip, y_R_hip)]
#
#             # Get Knees Shoulder Distance
#             knee_shoulder_distance = int((abs(diff_shoulder - diff_knees)/diff_knees)*100)
#
#             # Get Face Degree
#             degrees_ear_face = get_angle((rh_ear[0], rh_ear[1]), (lh_ear[0], lh_ear[1]))
#             # cv2.line(img_ori,(rh_ear[0], rh_ear[1]), (lh_ear[0], lh_ear[1]), (0, 0, 255), 3)
#
#             # Get Body Degree
#             degrees_body = get_angle(middle_shoulder_point, middle_hip_point)
#             # cv2.line(img_ori, middle_shoulder_point, middle_hip_point, (0, 0, 255), 3)
#
#             # Get Shoulders Degree
#             degrees_shoulder = get_angle((x_R_shoulder, y_R_shoulder), (x_L_shoulder, y_L_shoulder))
#             # cv2.line(img_ori, (x_R_shoulder, y_R_shoulder), (x_L_shoulder, y_L_shoulder), (0, 0, 255), 3)
#
#             # Get Length Shoulder
#             length_shoulder = abs( x_R_shoulder - x_L_shoulder)
#             length_shoulder_3times = length_shoulder * 3
#             length_shoulder_2times = length_shoulder * 2
#             length_half_shoulder = length_shoulder // 2
#             length_thirdhalf_shoulder = length_shoulder // 3
#
#             RH_Cropped, RArm_Cropped, LArm_Cropped, LH_Cropped, RH_Cropped_Coor, RArm_Cropped_Coor, LArm_Cropped_Coor, LH_Cropped_Coor = body_segment(img_ori, rh_elbow, rh_shoulder, rh_wrist,
#                                                                   rh_index_finger, lh_shoulder, lh_wrist, lh_elbow, length_shoulder,
#                                                                   length_shoulder_2times, length_shoulder_3times,
#                                                                   length_half_shoulder, length_thirdhalf_shoulder)
#
#     except Exception as e:
#         print('MediaPipe Failed:', e)
#         knee_shoulder_distance = 0
#         degree_lh_shoulder_elbow = 45
#         degrees_ear_face = 90
#         degrees_body = 90
#         degrees_shoulder = 0
#         rh_hip, lh_hip, lh_knees, rh_knees, lh_shoulder, rh_shoulder, lh_ear, rh_ear = [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]
#         lh_slope_value = 0.05
#         lh_finger_wrist_elbow_coor = [[(0, 0), (0, 0)], [(0, 0), (0, 0)], [(0, 0), (0, 0)]]
#         head_rectangle_coordinate, body_rectangle_coordinate = [(0, 0), (0, 0)], [(0, 0), (0, 0)]
#         RH_Cropped, RArm_Cropped, LArm_Cropped, LH_Cropped = image, image, image, image
#         RH_Cropped_Coor, RArm_Cropped_Coor, LArm_Cropped_Coor, LH_Cropped_Coor = [(0, 0), (0, 0)], [(0, 0), (0, 0)], [(0, 0), (0, 0)], [(0, 0), (0, 0)]
#     return img_ori, RH_Cropped, RArm_Cropped, LArm_Cropped, LH_Cropped, RH_Cropped_Coor, RArm_Cropped_Coor, LArm_Cropped_Coor, LH_Cropped_Coor, \
#            head_rectangle_coordinate, body_rectangle_coordinate, knee_shoulder_distance, degrees_ear_face, \
#            degrees_body, degrees_shoulder, degree_lh_shoulder_elbow, lh_slope_value, rh_hip, lh_hip, lh_finger_wrist_elbow_coor, \
#            lh_knees, rh_knees, lh_shoulder, rh_shoulder, lh_ear, rh_ear


def body_segment(frame, rh_elbow, rh_shoulder, rh_wrist, rh_index_finger, lh_shoulder, lh_wrist, lh_elbow, length_shoulder,
                length_shoulder_2times, length_shoulder_3times, length_half_shoulder, length_thirdhalf_shoulder):
    img = frame.copy()
    x_WRRH = rh_wrist[0]
    y_WRRH = rh_wrist[1]
    top_left_RH_box = (x_WRRH - (length_thirdhalf_shoulder), y_WRRH - (length_thirdhalf_shoulder//3))
    bottom_right_RH_box = (x_WRRH + (length_thirdhalf_shoulder), y_WRRH + (length_half_shoulder))
    RH_box = [top_left_RH_box, bottom_right_RH_box]

    x_WRLH = lh_wrist[0]
    y_WRLH = lh_wrist[1]
    top_left_LH_box = (x_WRLH - length_half_shoulder, y_WRLH - (length_thirdhalf_shoulder))
    bottom_right_LH_box = (x_WRLH + (length_thirdhalf_shoulder // 2), y_WRLH + (length_thirdhalf_shoulder // 2))
    LH_box = [top_left_LH_box, bottom_right_LH_box]

    x_RArm = rh_shoulder[0]
    y_RArm = rh_shoulder[1]
    top_left_RArm_box = (x_RArm - length_shoulder_2times, y_RArm - length_thirdhalf_shoulder)
    bottom_right_RArm_box = (x_RArm + length_half_shoulder, y_RArm + length_shoulder + length_half_shoulder)
    RArm_box = [top_left_RArm_box, bottom_right_RArm_box]

    x_LArm = lh_shoulder[0]
    y_LArm = lh_shoulder[1]
    top_left_LArm_box = (x_LArm - length_half_shoulder, y_LArm - length_thirdhalf_shoulder)
    bottom_left_LArm_box = (x_LArm + length_shoulder + length_half_shoulder, y_LArm + length_shoulder)
    LArm_box = [top_left_LArm_box, bottom_left_LArm_box]

    RH_Cropped = cropped_by_pixel(img, RH_box[0][0], RH_box[0][1], RH_box[1][0], RH_box[1][1])
    RArm_Cropped = cropped_by_pixel(img, RArm_box[0][0], RArm_box[0][1], RArm_box[1][0], RArm_box[1][1])
    LArm_Cropped = cropped_by_pixel(img, LArm_box[0][0], LArm_box[0][1], LArm_box[1][0], LArm_box[1][1])
    LH_Cropped = cropped_by_pixel(img, LH_box[0][0], LH_box[0][1], LH_box[1][0], LH_box[1][1])

    if rh_wrist[0] < rh_shoulder[0]:
        RArm_Cropped_Coor = [(rh_wrist[0], rh_shoulder[1]), (RArm_box[1][0], RArm_box[1][1])]
    else:
        RArm_Cropped_Coor = [(rh_elbow[0], rh_shoulder[1]), (RArm_box[1][0], RArm_box[1][1])]

    if lh_elbow[0] > lh_shoulder[0]:
        if lh_elbow[1] > lh_wrist[1]:
            LArm_Cropped_Coor = [(LArm_box[0][0], LArm_box[0][1]), (lh_elbow[0], lh_elbow[1])]
        else:
            LArm_Cropped_Coor = [(LArm_box[0][0], LArm_box[0][1]), (lh_elbow[0], lh_wrist[1])]
    else:
        LArm_Cropped_Coor = [(LArm_box[0][0], LArm_box[0][1]), (lh_shoulder[0], lh_elbow[1])]

    RH_Cropped_Coor = [(RH_box[0][0], RH_box[0][1]), (RH_box[1][0], RH_box[1][1])]
    LH_Cropped_Coor = [(LH_box[0][0], LH_box[0][1]), (LH_box[1][0], LH_box[1][1])]
    # RArm_Cropped_Coor = [(RArm_box[0][0], RArm_box[0][1]), (RArm_box[1][0], RArm_box[1][1])]
    # LArm_Cropped_Coor = [(LArm_box[0][0], LArm_box[0][1]), (LArm_box[1][0], LArm_box[1][1])]
    return RH_Cropped, RArm_Cropped, LArm_Cropped, LH_Cropped, RH_Cropped_Coor, RArm_Cropped_Coor, LArm_Cropped_Coor, LH_Cropped_Coor


def scaleIMG(img, scale = 1) :
    return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation = cv2.INTER_AREA) if scale else img

def distance2D(p1, p2) :
    distance = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    return distance

def erhu_bow_segment(frame, subtractor, skipFrame, horizontalLinePoint1, horizontalLinePoint2, verticalLinePoint1,
                     verticalLinePoint2, horizontalAvgSize, verticalAvgSize, horizontalArray1, horizontalArray2,
                     verticalArray1, verticalArray2, crossDistanceMaxH , crossDistanceMaxV, handPointsAvgSize,
                     rightHandPoints, leftHandPoints):

    scale = 600 / frame.shape[0]
    frame = scaleIMG(frame, scale)
    showLines = False
    # print('skip_frame:', skipFrame)
    img = frame.copy()

    minLineSize = img.shape[0] / 3

    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    poseResults = pose.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # mp_drawing.draw_landmarks(img, poseResults.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    bottomLimit = img.shape[0]
    rightShoulderX = 0

    if hasattr(poseResults.pose_landmarks, 'landmark'):
        handRight = [int(poseResults.pose_landmarks.landmark[20].x * img.shape[1]),
                     int(poseResults.pose_landmarks.landmark[16].y * img.shape[0])]
        handLeft = [int(poseResults.pose_landmarks.landmark[19].x * img.shape[1]),
                    int(poseResults.pose_landmarks.landmark[19].y * img.shape[0])]

        rightShoulderX = poseResults.pose_landmarks.landmark[12].x * img.shape[1]
        leftHipX = poseResults.pose_landmarks.landmark[23].x * img.shape[1] - 30
        bottomLimit = poseResults.pose_landmarks.landmark[23].y * img.shape[0] + 30

        if len(rightHandPoints) > handPointsAvgSize:
            rightHandPoints.pop(0)
        rightHandPoints.append(handRight)

        if len(leftHandPoints) > handPointsAvgSize:
            leftHandPoints.pop(0)
        leftHandPoints.append(handLeft)

    edges = cv2.Canny(img, 100, 50, apertureSize=3)
    noiseValue = np.mean(edges)
    # print(noiseValue)

    if noiseValue > 10:
        blur = cv2.blur(img, (10, 5))
        edges = cv2.Canny(blur, 50, 100, apertureSize=3)
        mask = subtractor.apply(blur)
    else:
        mask = subtractor.apply(img)

    edges = cv2.dilate(edges, np.ones((3, 2), np.uint8), iterations=1)
    # cv2.imshow("edges", edges)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8))
    # cv2.imshow("Mask", mask)

    edges[np.where((mask == 0))] = 0
    # cv2.imshow("edges", edges)

    _horizontalLinesPoint1 = []
    _horizontalLinesPoint2 = []
    _verticalLinesPoint1 = []
    _verticalLinesPoint2 = []
    bow_line_coor = [(0, 0), (0, 0)]
    erhu_line_coor = [(0, 0), (0, 0)]

    if skipFrame == 0 :
        lines = cv2.HoughLinesP(image=edges,rho=1, theta=np.pi/180, threshold=100, lines=np.array([]), minLineLength=minLineSize, maxLineGap=80)

        if hasattr(lines, 'shape'):
            a, b, c = lines.shape

            for i in range(a):
                x1, y1 = lines[i][0][0], lines[i][0][1]
                x2, y2 = lines[i][0][2], lines[i][0][3]
                # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)

                theta = math.atan2(y2 - y1, x2 - x1)
                angle = math.degrees(theta)
                if angle < 0:
                    angle = -angle
                # print(angle)

                if angle < 60:  # horizontal line
                    if x1 > x2:
                        x1, y1, x2, y2 = x2, y2, x1, y1

                    if y1 > frame.shape[0] / 2.2 and y2 > frame.shape[0] / 2.2 and x2 > rightShoulderX:
                        _horizontalLinesPoint1.append([x1, y1])
                        _horizontalLinesPoint2.append([x2, y2])

                        if showLines:
                            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)

                if angle > 70:  # vertical line vertical
                    if y1 > y2:
                        x1, y1, x2, y2 = x2, y2, x1, y1

                    dist = crossDistanceMaxV
                    if len(leftHandPoints) > 0:
                        leftHandPoint = np.mean(leftHandPoints, axis=0)

                        if showLines:
                            cv2.circle(img, (int(leftHandPoint[0]), int(leftHandPoint[1])), 5, (0, 255, 0), -1)

                        p1 = np.array([x1, y1])
                        p2 = np.array([x2, y2])
                        p3 = np.array(leftHandPoint)
                        dist = abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))
                    try:
                        if x1 > frame.shape[1] / 2.2 and x2 > frame.shape[
                            1] / 2.2 and dist < crossDistanceMaxV and x2 > leftHipX and y2 < bottomLimit:
                            _verticalLinesPoint1.append([x1, y1])
                            _verticalLinesPoint2.append([x2, y2])

                            if showLines:
                                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
                    except:
                        print('Missing Landmark')

        meanAngle = 0
        angles = []
        diffAngle = 180
        diffAngleArr = []
        meanDistance = 0
        distances = []

        if len(_verticalLinesPoint1) > 0:
            for i, point in enumerate(_verticalLinesPoint1):
                x1, y1 = _verticalLinesPoint1[i][0], _verticalLinesPoint1[i][1]
                x2, y2 = _verticalLinesPoint2[i][0], _verticalLinesPoint2[i][1]

                theta = math.atan2(y2 - y1, x2 - x1)
                angle = math.degrees(theta)
                angles.append(angle)

                distance = math.dist([x1, y1], [x2, y2])
                distances.append(distance)

            meanAngle = np.mean(angles, axis=0)
            meanDistance = np.mean(distances, axis=0)

            for angle in angles:
                diffAngleArr.append(abs(angle - meanAngle))
            diffAngle = np.mean(diffAngleArr, axis=0) * 1.1

            for i in range(len(angles) - 1, -1, -1):
                angle = angles[i]
                distance = distances[i]

                if len(_verticalLinesPoint1) > 1 and abs(
                        angle - meanAngle) > diffAngle or angle < meanAngle or distance < meanDistance:
                    _verticalLinesPoint1.pop(i)
                    _verticalLinesPoint2.pop(i)

        meanAngle = 0
        angles = []
        diffAngle = 180
        diffAngleArr = []
        meanDistance = 0
        distances = []
        meanLength = 0
        lengths = []

        if len(_horizontalLinesPoint1) > 0:
            if len(leftHandPoints) > 0:
                rightHandPoint = np.mean(rightHandPoints, axis=0)

                if showLines:
                    cv2.circle(img, (int(rightHandPoint[0]), int(rightHandPoint[1])), 5, (0, 0, 255), -1)

                for i, point in enumerate(_horizontalLinesPoint1):
                    x1, y1 = _horizontalLinesPoint1[i][0], _horizontalLinesPoint1[i][1]
                    x2, y2 = _horizontalLinesPoint2[i][0], _horizontalLinesPoint2[i][1]

                    theta = math.atan2(y2 - y1, x2 - x1)
                    angle = math.degrees(theta)
                    angles.append(angle)

                    distance = math.dist([x1, y1], rightHandPoint)
                    distances.append(distance)

                    length = math.dist([x1, y1], [x2, y2])
                    lengths.append(length)

                meanAngle = np.mean(angles, axis=0)
                meanDistance = np.mean(distances, axis=0) * 1.1
                meanLength = np.mean(lengths, axis=0) * 0.9

                for angle in angles:
                    diffAngleArr.append(abs(angle - meanAngle))
                diffAngle = np.mean(diffAngleArr, axis=0) * 1.1

                for i in range(len(angles) - 1, -1, -1):
                    angle = angles[i]
                    distance = distances[i]
                    length = lengths[i]

                    if len(_horizontalLinesPoint1) > 1 and abs(
                            angle - meanAngle) > diffAngle or distance > meanDistance or length < meanLength:
                        _horizontalLinesPoint1.pop(i)
                        _horizontalLinesPoint2.pop(i)

        if len(_horizontalLinesPoint1) > 0:
            horizontalLinePoint1 = np.mean(_horizontalLinesPoint1, axis=0)
            horizontalLinePoint2 = np.mean(_horizontalLinesPoint2, axis=0)

        if len(_verticalLinesPoint1) > 0:
            verticalLinePoint1 = np.mean(_verticalLinesPoint1, axis=0)
            verticalLinePoint2 = np.mean(_verticalLinesPoint2, axis=0)

        if len(horizontalLinePoint1) > 0:
            if len(horizontalArray1) > horizontalAvgSize:
                horizontalArray1.pop(0)
                horizontalArray2.pop(0)

            horizontalArray1.append(horizontalLinePoint1)
            horizontalArray2.append(horizontalLinePoint2)
            horizontalLinePoint1 = np.mean(horizontalArray1, axis=0)
            horizontalLinePoint2 = np.mean(horizontalArray2, axis=0)

            x1, y1 = int(horizontalLinePoint1[0]), int(horizontalLinePoint1[1])
            x2, y2 = int(horizontalLinePoint2[0]), int(horizontalLinePoint2[1])
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
            bow_line_coor = [(x1, y1), (x2, y2)]
        else:
            try:
                horizontalLinePoint1 = np.mean(horizontalArray1, axis=0)
                horizontalLinePoint2 = np.mean(horizontalArray2, axis=0)
                x1, y1 = int(horizontalLinePoint1[0]), int(horizontalLinePoint1[1])
                x2, y2 = int(horizontalLinePoint2[0]), int(horizontalLinePoint2[1])
                bow_line_coor = [(x1, y1), (x2, y2)]
            except:
                bow_line_coor = [(0, 0),(0, 0)]

        if len(verticalLinePoint1) > 0 :
            if len(verticalArray1) > verticalAvgSize:
                verticalArray1.pop(0)
                verticalArray2.pop(0)

            verticalArray1.append(verticalLinePoint1)
            verticalArray2.append(verticalLinePoint2)
            verticalLinePoint1 = np.mean(verticalArray1, axis=0)
            verticalLinePoint2 = np.mean(verticalArray2, axis=0)

            x1, y1 = int(verticalLinePoint1[0]), int(verticalLinePoint1[1])
            x2, y2 = int(verticalLinePoint2[0]), int(verticalLinePoint2[1])
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
            erhu_line_coor = [(x1, y1), (x2, y2)]
        else:
            try:
                verticalLinePoint1 = np.mean(verticalArray1, axis=0)
                verticalLinePoint2 = np.mean(verticalArray2, axis=0)
                x1, y1 = int(verticalLinePoint1[0]), int(verticalLinePoint1[1])
                x2, y2 = int(verticalLinePoint2[0]), int(verticalLinePoint2[1])
                erhu_line_coor = [(x1, y1), (x2, y2)]
            except:
                erhu_line_coor = [(0, 0),(0, 0)]
    else :
        skipFrame -= 1

    return skipFrame, erhu_line_coor, bow_line_coor

def body_segment_mmpose(frame):
    img = frame.copy()
    mmdet_results = inference_detector(det_model, img)
    person_results = process_mmdet_results(mmdet_results, 1)

    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        img,
        person_results,
        bbox_thr=thresHold,
        format='xyxy',
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=return_heatmap,
        outputs=output_layer_names)

    vis_img = vis_pose_result(
        pose_model,
        img,
        pose_results,
        dataset=dataset,
        dataset_info=dataset_info,
        kpt_score_thr=thresHold,
        bbox_color='white',
        radius=3,
        show=False)

    # cv2.imshow('mmpose', img)
    # cv2.waitKey(0)
    # exit()

    image_height, image_width, _ = img.shape

    x_WRLH = int(pose_results[0]['keypoints'][9][0])
    y_WRLH = int(pose_results[0]['keypoints'][9][1])
    lh_wrist = [x_WRLH, y_WRLH]

    x_WRRH = int(pose_results[0]['keypoints'][117][0])
    y_WRRH = int(pose_results[0]['keypoints'][117][1])
    rh_wrist = [x_WRRH, y_WRRH]

    x_MIDFGRLH = int(pose_results[0]['keypoints'][102][0])
    y_MIDFGRLH = int(pose_results[0]['keypoints'][102][1])
    lh_middle_finger = [x_MIDFGRLH, y_MIDFGRLH]

    x_MIDFGRLH2 = int(pose_results[0]['keypoints'][100][0])
    y_MIDFGRLH2 = int(pose_results[0]['keypoints'][100][1])
    lh_middle_finger2 = [x_MIDFGRLH2, y_MIDFGRLH2]

    x_LTFGRRH = int(pose_results[0]['keypoints'][132][0])
    y_LTFGRRH = int(pose_results[0]['keypoints'][132][1])

    # Get length of shoulder
    x_R_shoulder = int(pose_results[0]['keypoints'][6][0])
    y_R_shoulder = int(pose_results[0]['keypoints'][6][1])
    rh_shoulder = [x_R_shoulder, y_R_shoulder]

    x_L_shoulder = int(pose_results[0]['keypoints'][5][0])
    y_L_shoulder = int(pose_results[0]['keypoints'][5][1])
    lh_shoulder = [x_L_shoulder, y_L_shoulder]
    diff_shoulder = abs(x_L_shoulder - x_R_shoulder)
    # Get right arm box
    x_RArm = int(pose_results[0]['keypoints'][6][0])
    y_RArm = int(pose_results[0]['keypoints'][6][1])
    x_RElbow = int(pose_results[0]['keypoints'][8][0])
    x_LArm = int(pose_results[0]['keypoints'][5][0])
    y_LArm = int(pose_results[0]['keypoints'][5][1])

    # Get Hip
    x_R_hip = int(pose_results[0]['keypoints'][12][0])
    y_R_hip = int(pose_results[0]['keypoints'][12][1])
    x_L_hip = int(pose_results[0]['keypoints'][11][0])
    y_L_hip = int(pose_results[0]['keypoints'][11][1])
    rh_hip = [x_R_hip, y_R_hip]
    lh_hip = [x_L_hip, y_L_hip]

    # Get Knees
    x_L_knees = int(pose_results[0]['keypoints'][13][0])
    y_L_knees = int(pose_results[0]['keypoints'][13][1])
    x_R_knees = int(pose_results[0]['keypoints'][14][0])
    y_R_knees = int(pose_results[0]['keypoints'][14][1])
    lh_knees = [x_L_knees, y_L_knees]
    rh_knees = [x_R_knees, y_R_knees]
    diff_knees = abs(x_L_knees - x_R_knees)

    #Get Eye
    x_L_eye = int(pose_results[0]['keypoints'][1][0]) + (abs(x_R_shoulder - x_L_shoulder) // 4)
    y_L_eye = int(pose_results[0]['keypoints'][1][1]) - (abs(x_R_shoulder - x_L_shoulder) // 4)
    x_R_eye = int(pose_results[0]['keypoints'][2][0]) - (abs(x_R_shoulder - x_L_shoulder) // 4)
    y_R_eye = int(pose_results[0]['keypoints'][2][1]) + (abs(x_R_shoulder - x_L_shoulder) // 4)

    # Get Elbow Left Hand
    x_L_elbow = int(pose_results[0]['keypoints'][7][0])
    y_L_elbow = int(pose_results[0]['keypoints'][7][1])
    lh_elbow = [x_L_elbow, y_L_elbow]

    # Get Ear
    x_R_ear = int(pose_results[0]['keypoints'][4][0])
    y_R_ear = int(pose_results[0]['keypoints'][4][1])
    x_L_ear = int(pose_results[0]['keypoints'][3][0])
    y_L_ear = int(pose_results[0]['keypoints'][3][1])
    lh_ear = [x_L_ear, y_L_ear]
    rh_ear = [x_R_ear, y_R_ear]

    # Get Middle Body
    if x_R_shoulder > x_L_shoulder:
        middle_shoulder_point = ((abs(x_R_shoulder - x_L_shoulder) // 2) + x_L_shoulder, y_L_shoulder)
    else:
        middle_shoulder_point = ((abs(x_L_shoulder - x_R_shoulder) // 2) + x_R_shoulder, y_R_shoulder)
    if x_R_hip > x_L_hip:
        middle_hip_point = (abs((x_R_hip - x_L_hip) // 2) + x_L_hip, y_L_hip)
    else:
        middle_hip_point = (abs((x_L_hip - x_R_hip) // 2) + x_R_hip, y_R_hip)

    # Get Finger Wrist Slope Left Hand Middle Finger
    lh_finger_wrist_degree = get_angle(lh_middle_finger2, lh_wrist)

    # Get Wrist Elbow Slope Left Hand Elbow
    lh_wrist_elbow_degree = get_angle(lh_wrist, lh_elbow)

    head_rectangle_coordinate = [(x_L_eye, y_L_eye), (x_R_eye, y_R_eye)]
    body_rectangle_coordinate = [(x_L_shoulder, y_L_shoulder), (x_R_hip, y_R_hip)]
    if diff_knees == 0: diff_knees = 1
    knee_shoulder_distance = int((abs(diff_shoulder - diff_knees) / diff_knees) * 100)
    degrees_ear_face = get_angle((rh_ear[0], rh_ear[1]), (lh_ear[0], lh_ear[1]))
    degrees_body = get_angle(middle_shoulder_point, middle_hip_point)
    degrees_shoulder = get_angle((x_R_shoulder, y_R_shoulder), (x_L_shoulder, y_L_shoulder))
    degree_lh_shoulder_elbow = get_angle(lh_shoulder, lh_elbow)
    lh_slope_value = abs(lh_wrist_elbow_degree - lh_finger_wrist_degree)
    lh_finger_wrist_elbow_coor = [lh_middle_finger, lh_middle_finger2, lh_wrist, lh_elbow]
    # if frame_number == 1 :
    # print('frame_number', frame_number)
    length_shoulders = abs(x_R_shoulder - x_L_shoulder)
    length_shoulders_3times = length_shoulders * 3
    length_shoulders_2times = length_shoulders * 2
    length_half_shoulders = length_shoulders // 2
    length_thirdhalf_shoulders = length_half_shoulders // 3

    return length_shoulders,length_half_shoulders, length_shoulders_3times,length_shoulders_2times, length_thirdhalf_shoulders, \
           x_WRRH, y_WRRH, x_LTFGRRH, y_LTFGRRH, x_RElbow, y_RArm, x_RArm, x_LArm, y_LArm, x_WRLH, y_WRLH, \
           head_rectangle_coordinate, body_rectangle_coordinate, knee_shoulder_distance, degrees_ear_face,\
           degrees_body, degrees_shoulder, degree_lh_shoulder_elbow, lh_slope_value, rh_hip, lh_hip, lh_finger_wrist_elbow_coor,\
           lh_knees, rh_knees, lh_shoulder, rh_shoulder, lh_ear, rh_ear


def get_cropped_image(frame, length_shoulders,length_half_shoulders, length_shoulders_3times,length_shoulders_2times, length_thirdhalf_shoulders, \
           x_WRRH, y_WRRH, x_LTFGRRH, y_LTFGRRH, x_RElbow, y_RArm, x_RArm, x_LArm, y_LArm, x_WRLH, y_WRLH, \
           head_rectangle_coordinate, body_rectangle_coordinate, knee_shoulder_distance, degrees_ear_face,\
           degrees_body, degrees_shoulder, degree_lh_shoulder_elbow, lh_slope_value, rh_hip, lh_hip, lh_finger_wrist_elbow_coor,\
           lh_knees, rh_knees, lh_shoulder, rh_shoulder, lh_ear, rh_ear):
    img = frame.copy()

    top_left_LH_box = (x_WRLH - int(length_thirdhalf_shoulders*3), y_WRLH - int(length_thirdhalf_shoulders*2))
    bottom_right_LH_box = (x_WRLH + int(length_thirdhalf_shoulders*0.5), y_WRLH + int(length_thirdhalf_shoulders*0.5))
    LH_box = [top_left_LH_box, bottom_right_LH_box]

    top_left_RH_box = (x_WRRH - length_thirdhalf_shoulders, y_WRRH - length_thirdhalf_shoulders)
    bottom_right_RH_box = (x_WRRH + length_thirdhalf_shoulders + length_thirdhalf_shoulders, y_WRRH + length_thirdhalf_shoulders)
    RH_box = [top_left_RH_box, bottom_right_RH_box]

    top_left_RArm_box = (x_RArm - length_shoulders_2times, y_RArm - length_thirdhalf_shoulders)
    bottom_right_RArm_box = (x_RArm + length_half_shoulders, y_RArm + length_shoulders + length_half_shoulders)
    RArm_box = [top_left_RArm_box, bottom_right_RArm_box]

    top_left_LArm_box = (x_LArm - length_half_shoulders, y_LArm - length_thirdhalf_shoulders)
    bottom_left_LArm_box = (x_LArm + length_shoulders + length_half_shoulders, y_LArm + length_shoulders + length_thirdhalf_shoulders)
    LArm_box = [top_left_LArm_box, bottom_left_LArm_box]
    RH_Cropped = cropped_by_pixel(img, RH_box[0][0], RH_box[0][1], RH_box[1][0], RH_box[1][1])
    RH_Cropped_Coor = [(RH_box[0][0], RH_box[0][1]), (RH_box[1][0], RH_box[1][1])]
    LH_Cropped = cropped_by_pixel(img, LH_box[0][0], LH_box[0][1], LH_box[1][0], LH_box[1][1])
    LH_Cropped_Coor = [(LH_box[0][0], LH_box[0][1]),(LH_box[1][0], LH_box[1][1])]
    RArm_Cropped = cropped_by_pixel(img, RArm_box[0][0], RArm_box[0][1], RArm_box[1][0], RArm_box[1][1])
    RArm_Cropped_Coor = [(RArm_box[0][0], RArm_box[0][1]), (RArm_box[1][0], RArm_box[1][1])]
    LArm_Cropped = cropped_by_pixel(img, LArm_box[0][0], LArm_box[0][1], LArm_box[1][0], LArm_box[1][1])
    LArm_Cropped_Coor = [(LArm_box[0][0], LArm_box[0][1]),(LArm_box[1][0], LArm_box[1][1])]

    return img, RH_Cropped, RArm_Cropped, LArm_Cropped, LH_Cropped, \
           RH_Cropped_Coor, RArm_Cropped_Coor, LArm_Cropped_Coor, LH_Cropped_Coor, \
           head_rectangle_coordinate, body_rectangle_coordinate, knee_shoulder_distance, degrees_ear_face, \
           degrees_body, degrees_shoulder, degree_lh_shoulder_elbow, lh_slope_value, rh_hip, lh_hip, lh_finger_wrist_elbow_coor, \
           lh_knees, rh_knees, lh_shoulder, rh_shoulder, lh_ear, rh_ear

def getKeypoints(frame):
    img = frame.copy()
    mmdet_results = inference_detector(det_model, img)
    person_results = process_mmdet_results(mmdet_results, 1)
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        img,
        person_results,
        bbox_thr=thresHold,
        format='xyxy',
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=return_heatmap,
        outputs=output_layer_names)

    vis_img = vis_pose_result(
        pose_model,
        img,
        pose_results,
        dataset=dataset,
        dataset_info=dataset_info,
        kpt_score_thr=thresHold,
        bbox_color='white',
        radius=8,
        show=False)
    # print('Keypoints', pose_results[0]['keypoints'][112])
    key_112_x = int(pose_results[0]['keypoints'][112][0])
    key_112_y = int(pose_results[0]['keypoints'][112][1])
    key_113_x = int(pose_results[0]['keypoints'][113][0])
    key_113_y = int(pose_results[0]['keypoints'][113][1])
    key_114_x = int(pose_results[0]['keypoints'][114][0])
    key_114_y = int(pose_results[0]['keypoints'][114][1])
    key_115_x = int(pose_results[0]['keypoints'][115][0])
    key_115_y = int(pose_results[0]['keypoints'][115][1])
    key_116_x = int(pose_results[0]['keypoints'][116][0])
    key_116_y = int(pose_results[0]['keypoints'][116][1])
    key_117_x = int(pose_results[0]['keypoints'][117][0])
    key_117_y = int(pose_results[0]['keypoints'][117][1])
    key_118_x = int(pose_results[0]['keypoints'][118][0])
    key_118_y = int(pose_results[0]['keypoints'][118][1])
    key_119_x = int(pose_results[0]['keypoints'][119][0])
    key_119_y = int(pose_results[0]['keypoints'][119][1])
    key_120_x = int(pose_results[0]['keypoints'][120][0])
    key_120_y = int(pose_results[0]['keypoints'][120][1])
    key_121_x = int(pose_results[0]['keypoints'][121][0])
    key_121_y = int(pose_results[0]['keypoints'][121][1])
    key_122_x = int(pose_results[0]['keypoints'][122][0])
    key_122_y = int(pose_results[0]['keypoints'][122][1])
    key_123_x = int(pose_results[0]['keypoints'][123][0])
    key_123_y = int(pose_results[0]['keypoints'][123][1])
    key_124_x = int(pose_results[0]['keypoints'][124][0])
    key_124_y = int(pose_results[0]['keypoints'][124][1])
    key_125_x = int(pose_results[0]['keypoints'][125][0])
    key_125_y = int(pose_results[0]['keypoints'][125][1])
    key_126_x = int(pose_results[0]['keypoints'][126][0])
    key_126_y = int(pose_results[0]['keypoints'][126][1])
    key_127_x = int(pose_results[0]['keypoints'][127][0])
    key_127_y = int(pose_results[0]['keypoints'][127][1])
    key_128_x = int(pose_results[0]['keypoints'][128][0])
    key_128_y = int(pose_results[0]['keypoints'][128][1])
    key_129_x = int(pose_results[0]['keypoints'][129][0])
    key_129_y = int(pose_results[0]['keypoints'][129][1])
    key_130_x = int(pose_results[0]['keypoints'][130][0])
    key_130_y = int(pose_results[0]['keypoints'][130][1])
    key_131_x = int(pose_results[0]['keypoints'][131][0])
    key_131_y = int(pose_results[0]['keypoints'][131][1])
    key_132_x = int(pose_results[0]['keypoints'][132][0])
    key_132_y = int(pose_results[0]['keypoints'][132][1])
    keypoints_rh = [(key_112_x, key_112_y), (key_113_x, key_113_y), (key_114_x, key_114_y), (key_115_x, key_115_y),
                    (key_116_x, key_116_y), (key_117_x, key_117_y), (key_118_x, key_118_y), (key_119_x, key_119_y),
                    (key_120_x, key_120_y), (key_121_x, key_121_y), (key_122_x, key_122_y), (key_123_x, key_123_y),
                    (key_124_x, key_124_y), (key_125_x, key_125_y), (key_126_x, key_126_y), (key_127_x, key_127_y),
                    (key_128_x, key_128_y), (key_129_x, key_129_y), (key_130_x, key_130_y), (key_131_x, key_131_y),
                    (key_132_x, key_132_y)]
    key_91_x = int(pose_results[0]['keypoints'][91][0])
    key_91_y = int(pose_results[0]['keypoints'][91][1])
    key_92_x = int(pose_results[0]['keypoints'][92][0])
    key_92_y = int(pose_results[0]['keypoints'][92][1])
    key_93_x = int(pose_results[0]['keypoints'][93][0])
    key_93_y = int(pose_results[0]['keypoints'][93][1])
    key_94_x = int(pose_results[0]['keypoints'][94][0])
    key_94_y = int(pose_results[0]['keypoints'][94][1])
    key_95_x = int(pose_results[0]['keypoints'][95][0])
    key_95_y = int(pose_results[0]['keypoints'][95][1])
    key_96_x = int(pose_results[0]['keypoints'][96][0])
    key_96_y = int(pose_results[0]['keypoints'][96][1])
    key_97_x = int(pose_results[0]['keypoints'][97][0])
    key_97_y = int(pose_results[0]['keypoints'][97][1])
    key_98_x = int(pose_results[0]['keypoints'][98][0])
    key_98_y = int(pose_results[0]['keypoints'][98][1])
    key_99_x = int(pose_results[0]['keypoints'][99][0])
    key_99_y = int(pose_results[0]['keypoints'][99][1])
    key_100_x = int(pose_results[0]['keypoints'][100][0])
    key_100_y = int(pose_results[0]['keypoints'][100][1])
    key_101_x = int(pose_results[0]['keypoints'][101][0])
    key_101_y = int(pose_results[0]['keypoints'][101][1])
    key_102_x = int(pose_results[0]['keypoints'][102][0])
    key_102_y = int(pose_results[0]['keypoints'][102][1])
    key_103_x = int(pose_results[0]['keypoints'][103][0])
    key_103_y = int(pose_results[0]['keypoints'][103][1])
    key_104_x = int(pose_results[0]['keypoints'][104][0])
    key_104_y = int(pose_results[0]['keypoints'][104][1])
    key_105_x = int(pose_results[0]['keypoints'][105][0])
    key_105_y = int(pose_results[0]['keypoints'][105][1])
    key_106_x = int(pose_results[0]['keypoints'][106][0])
    key_106_y = int(pose_results[0]['keypoints'][106][1])
    key_107_x = int(pose_results[0]['keypoints'][107][0])
    key_107_y = int(pose_results[0]['keypoints'][107][1])
    key_108_x = int(pose_results[0]['keypoints'][108][0])
    key_108_y = int(pose_results[0]['keypoints'][108][1])
    key_109_x = int(pose_results[0]['keypoints'][109][0])
    key_109_y = int(pose_results[0]['keypoints'][109][1])
    key_110_x = int(pose_results[0]['keypoints'][110][0])
    key_110_y = int(pose_results[0]['keypoints'][110][1])
    key_111_x = int(pose_results[0]['keypoints'][111][0])
    key_111_y = int(pose_results[0]['keypoints'][111][1])
    keypoints_lh = [(key_91_x, key_91_y), (key_92_x, key_92_y), (key_93_x, key_93_y), (key_94_x, key_94_y),
                    (key_95_x, key_95_y), (key_96_x, key_96_y), (key_97_x, key_97_y), (key_98_x, key_98_y),
                    (key_99_x, key_99_y), (key_100_x, key_100_y), (key_101_x, key_101_y), (key_102_x, key_102_y),
                    (key_103_x, key_103_y), (key_104_x, key_104_y), (key_105_x, key_105_y), (key_106_x, key_106_y),
                    (key_107_x, key_107_y), (key_108_x, key_108_y), (key_109_x, key_109_y), (key_110_x, key_110_y),
                    (key_111_x, key_111_y)]
    key_7_x = int(pose_results[0]['keypoints'][7][0])
    key_7_y = int(pose_results[0]['keypoints'][7][1])
    key_9_x = int(pose_results[0]['keypoints'][9][0])
    key_9_y = int(pose_results[0]['keypoints'][9][1])
    key_100_x = int(pose_results[0]['keypoints'][100][0])
    key_100_y = int(pose_results[0]['keypoints'][100][1])
    key_103_x = int(pose_results[0]['keypoints'][103][0])
    key_103_y = int(pose_results[0]['keypoints'][103][1])
    keypoints_elbow = [(key_7_x, key_7_y), (key_9_x, key_9_y), (key_100_x, key_100_y), (key_103_x, key_103_y)]
    return keypoints_rh, keypoints_lh, keypoints_elbow
    # keypoint_list.append(pose_results[0]['keypoints'])
    # if cv2.waitKey(5) & 0xFF == 27:
    #     exit


def main_predict(video_input, isFlip = True):
    print('YOLO-GCN')
    print('video_input', video_input)
    start_now = datetime.now()
    start_time = start_now.strftime("%H:%M:%S")
    curr_date = str(start_now.day) + '-' + str(start_now.month) + '-' + str(start_now.year)
    curr_time = str(start_now.strftime('%H%M%S'))

    # # # model_left_hand = load_model(os.path.join(thisfolder, 'model/dataset_ver5/final/models_leftHand/model-007.h5'))
    # # model_left_arm = load_model(os.path.join(thisfolder, 'model/PredictionModel/model_left_arm/model-009.h5'))
    # # model_right_hand = load_model(os.path.join(thisfolder, 'model/PredictionModel/model_right_hand/model-018.h5'))
    # # model_right_arm = load_model(os.path.join(thisfolder, 'model/PredictionModel/model_right_arm/model-015.h5'))
    # # # model_left = load_model('/home/minelab/dev/erhu-project/classifier/model/PredictionModel/model_left/model-008.h5')
    # # # model_right = load_model('/home/minelab/dev/erhu-project/classifier/model/PredictionModel/model_right/model-014.h5')
    # # # model_left.summary()
    # # # model_right.summary()
    # # # model_left_hand.summary()
    # model_left_arm = load_model(os.path.join(thisfolder, 'model/PredictionModel/0409_0503_5th_7th_dataset_w_stride/models_dataset_leftArm/model-006.h5'))
    # model_right_hand = load_model(os.path.join(thisfolder, 'model/PredictionModel/0409_0503_5th_7th_dataset_w_stride/models_dataset_rightHand/model-008.h5'))
    # model_right_arm = load_model(os.path.join(thisfolder, 'model/PredictionModel/0409_0503_5th_7th_dataset_w_stride/models_dataset_rightArm/model-013.h5'))
    # model_left_arm.summary()
    # model_right_hand.summary()
    # model_right_arm.summary()
    # # print('TEST')
    # # exit()
    model = load_model("/home/minelab/dev/erhu-project/classifier/model/YOLO_GCN/rightHand/combine_new_model_right.h5", custom_objects={"GraphConv": GraphConv, "GraphMaxPool": GraphMaxPool})
    result_folder = os.path.join(os.path.abspath(__file__ + "/../../"), "predict")
    if os.path.exists(result_folder) == False:
        os.mkdir(result_folder)

    # dataset_folder  = 'video/5th_dataset_test'
    # dataset         = getImages(dataset_folder)
    # video_path      = 'video/5th_datasets_10 Apr 2022/0409 Separate each group/E23.mp4'
    video_path = video_input
    filename = video_path.split("/")[-1]
    filename = filename[:-4]
    print(video_path)
    # path = os.path.join(os.path.abspath(__file__ + "/../../"), "upload", video_path)

    videoInput = cv2.VideoCapture(video_path)
    LABELS = ["body", "bow", "erhu"]
    segment_image = custom_segmentation()
    segment_image.inferConfig(num_classes=3, class_names=LABELS)
    # segment_image.load_model(os.path.join(thisfolder, "model/SegmentationModel/4_5_dataset_13032022/mask_rcnn_model.081-0.129956.h5"))
    segment_image.load_model( os.path.join(thisfolder, "model/SegmentationModel/MixAllDataset/mask_rcnn_model.097-0.444332.h5"))
    properties = getVideoProperties(videoInput)
    # videoSegmentation = {}
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_fps = properties['FPS']
    # vid_fps = 30.0
    vid_width = properties['Width']
    vid_height = properties['Height']
    videoOut_1 = cv2.VideoWriter(os.path.join(result_folder, filename + ".mp4"), fourcc, vid_fps, (vid_width, vid_height))
    videoOut_1.set(cv2.CAP_PROP_FPS, 30.0)
    videoOut_2 = cv2.VideoWriter(os.path.join(result_folder, filename + ".mp4"), fourcc, vid_fps, (vid_width, vid_height))
    videoOut_2.set(cv2.CAP_PROP_FPS, 30.0)
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
    x_test_left_hand_point = []
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

    length = int(videoInput.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'vid length : {length}')

    start_frame_number = 0
    videoInput.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
    # videoInput.set(cv2.CAP_PROP_FPS, 30.0)

    frame_count = 1
    end_frame = length - 45
    idx_write_frame = 0
    print(videoInput.isOpened())

    # Variabel for Erhu and Bow Segment =============================
    subtractor = cv2.createBackgroundSubtractorKNN(10, 50, True)
    skipFrame = 10
    horizontalLinePoint1 = []
    horizontalLinePoint2 = []
    verticalLinePoint1 = []
    verticalLinePoint2 = []

    horizontalAvgSize = 5
    verticalAvgSize = 10
    horizontalArray1 = []
    horizontalArray2 = []
    verticalArray1 = []
    verticalArray2 = []
    crossDistanceMaxH = 10
    crossDistanceMaxV = 10
    handPointsAvgSize = 2
    rightHandPoints = []
    leftHandPoints = []
    # ================================================================
    frame_failed = 0
    output_json_number = 0
    get_scale = False
    length_shoulders = 0
    length_half_shoulders = 0
    length_shoulders_3times = 0
    length_shoulders_2times = 0
    length_thirdhalf_shoulders = 0
    x_WRRH = 0
    y_WRRH = 0
    x_LTFGRRH = 0
    y_LTFGRRH = 0
    x_RElbow = 0
    y_RArm = 0
    x_RArm = 0
    x_LArm = 0
    y_LArm = 0
    x_WRLH = 0
    y_WRLH = 0
    knee_shoulder_distance = 0
    degree_lh_shoulder_elbow = 45
    degrees_ear_face = 90
    degrees_body = 90
    degrees_shoulder = 0
    rh_hip, lh_hip, lh_knees, rh_knees, lh_shoulder, rh_shoulder, lh_ear, rh_ear = [0, 0], [0, 0], [0, 0], [0, 0], [0,0], [0, 0], [0, 0], [0, 0]
    lh_slope_value = 0.05
    lh_finger_wrist_elbow_coor = [[(0, 0), (0, 0)], [(0, 0), (0, 0)], [(0, 0), (0, 0)]]
    head_rectangle_coordinate, body_rectangle_coordinate = [(0, 0), (0, 0)], [(0, 0), (0, 0)]
    x_test_left_hand_keypoint = []
    x_test_right_hand_keypoint = []
    x_test_left_arm_keypoint = []
    hand_edge = np.zeros((21, 21))
    hand_edge2 = np.zeros((21, 21))

    hand_edge2[0, 0] = 1
    hand_edge2[1, 1] = 1
    hand_edge2[2, 2] = 1
    hand_edge2[3, 3] = 1
    hand_edge2[4, 4] = 1
    hand_edge2[5, 5] = 1
    hand_edge2[6, 6] = 1
    hand_edge2[7, 7] = 1
    hand_edge2[8, 8] = 1
    hand_edge2[9, 9] = 1
    hand_edge2[10, 10] = 1
    hand_edge2[11, 11] = 1
    hand_edge2[12, 12] = 1
    hand_edge2[13, 13] = 1
    hand_edge2[14, 14] = 1
    hand_edge2[15, 15] = 1
    hand_edge2[16, 16] = 1
    hand_edge2[17, 17] = 1
    hand_edge2[18, 18] = 1
    hand_edge2[19, 19] = 1
    hand_edge2[20, 20] = 1

    hand_edge2[0, 1] = 1
    hand_edge2[1, 2] = 1
    hand_edge2[2, 3] = 1
    hand_edge2[3, 4] = 1

    hand_edge2[0, 5] = 1
    hand_edge2[5, 6] = 1
    hand_edge2[6, 7] = 1
    hand_edge2[7, 8] = 1

    hand_edge2[0, 9] = 1
    hand_edge2[9, 10] = 1
    hand_edge2[10, 11] = 1
    hand_edge2[11, 12] = 1

    hand_edge2[1, 13] = 1
    hand_edge[13, 14] = 1
    hand_edge[14, 15] = 1
    hand_edge2[15, 16] = 1

    hand_edge2[1, 17] = 1
    hand_edge2[17, 18] = 1
    hand_edge2[18, 19] = 1
    hand_edge2[19, 20] = 1
    while videoInput.isOpened():
        # print('Frame open')
        success, frame = videoInput.read()
        if frame is None:
            break
        scale = 600 / frame.shape[0]
        frame = scaleIMG(frame, scale)
        err_mess_frame = [frame.shape[1] // 5, frame.shape[0] // 3]
        result_font = ImageFont.truetype(os.path.join(thisfolder, 'util/simsun.ttc'), err_mess_frame[0] // 12)

        # if get_scale == False:
        try:
            length_shoulders, length_half_shoulders, length_shoulders_3times, length_shoulders_2times, length_thirdhalf_shoulders, \
            x_WRRH, y_WRRH, x_LTFGRRH, y_LTFGRRH, x_RElbow, y_RArm, x_RArm, x_LArm, y_LArm, x_WRLH, y_WRLH, \
            head_rectangle_coordinate, body_rectangle_coordinate, knee_shoulder_distance, degrees_ear_face, \
            degrees_body, degrees_shoulder, degree_lh_shoulder_elbow, lh_slope_value, rh_hip, lh_hip, lh_finger_wrist_elbow_coor, \
            lh_knees, rh_knees, lh_shoulder, rh_shoulder, lh_ear, rh_ear = body_segment_mmpose(frame.copy())

            image, RH_Cropped, RArm_Cropped, LArm_Cropped, LH_Cropped, \
            RH_Cropped_Coor, RArm_Cropped_Coor, LArm_Cropped_Coor, LH_Cropped_Coor, \
            head_coordinate, body_coordinate, knees_shoulder_distance, degree_ear_face, \
            degree_body, degree_shoulder, degree_lh_shoulder_elbow, lh_slope_value, rh_hip, lh_hip, lh_finger_wrist_elbow_coor,\
            lh_knees, rh_knees, lh_shoulder, rh_shoulder, lh_ear, rh_ear = get_cropped_image(frame.copy(),
            length_shoulders,length_half_shoulders, length_shoulders_3times,length_shoulders_2times, length_thirdhalf_shoulders,
               x_WRRH, y_WRRH, x_LTFGRRH, y_LTFGRRH, x_RElbow, y_RArm, x_RArm, x_LArm, y_LArm, x_WRLH, y_WRLH,
               head_rectangle_coordinate, body_rectangle_coordinate, knee_shoulder_distance, degrees_ear_face,
               degrees_body, degrees_shoulder, degree_lh_shoulder_elbow, lh_slope_value, rh_hip, lh_hip, lh_finger_wrist_elbow_coor,
               lh_knees, rh_knees, lh_shoulder, rh_shoulder, lh_ear, rh_ear)
        except:
            print('No Landmark')
            continue

        # cv2.imshow('RH_Cropped', RH_Cropped)
        # cv2.imshow('RArm_Cropped', RArm_Cropped)
        # cv2.imshow('LArm_Cropped', LArm_Cropped)
        # cv2.imshow('LH_Cropped', LH_Cropped)
        rightHand_keypoints, leftHand_keypoints, leftArm_keypoint = getKeypoints(frame.copy())
        # print('rightHand_keypoints', rightHand_keypoints)
        # print('leftHand_keypoints', leftHand_keypoints)
        # print('leftArm_keypoint', leftArm_keypoint)
        cv2.waitKey(1)
        # exit()
        # for key in rightHand_keypoints:
        #     cv2.circle(image, key, 5, (0, 0, 255), thickness=-1)
        # for key in rightHand_keypoints:
        #     cv2.circle(image, key, 5, (0, 0, 255), thickness=-1)
        # cv2.imshow('Tracking', image)
        # cv2.waitKey(0)
        # exit()

        if videoInput.get(cv2.CAP_PROP_POS_FRAMES) == 1:
            videoOut_1 = cv2.VideoWriter(os.path.join(result_folder, filename + ".mp4"), fourcc, vid_fps,
                                         (frame.shape[1], frame.shape[0]))
            videoOut_1.set(cv2.CAP_PROP_FPS, 30.0)
            videoOut_2 = cv2.VideoWriter(os.path.join(result_folder, filename + "_err_msg.mp4"), fourcc, vid_fps,
                                         (frame.shape[1], frame.shape[0]))
            videoOut_2.set(cv2.CAP_PROP_FPS, 30.0)
        # cv2.imshow('test', frame)
        # cv2.waitKey(0)
        # image, rightHandPart, rightArmPart, leftHandPart, leftArmPart, \
        # halfBodyPart, leftArmPartOri, rightArmPartOri, leftHandPartOri, \
        # rightHandPartOri, left_arm_point, right_arm_point, right_hand_point, head_coordinate, body_coordinate, \
        # knees_shoulder_distance, degree_ear_face, degree_body, degree_shoulder, hip_left, hip_right = new_body_segment(
        #     frame.copy())
        # image, RH_Cropped, RArm_Cropped, LArm_Cropped, LH_Cropped, RH_Cropped_Coor, RArm_Cropped_Coor, LArm_Cropped_Coor, LH_Cropped_Coor, \
        # head_coordinate, body_coordinate, knees_shoulder_distance, degree_ear_face, \
        # degree_body, degree_shoulder, degree_lh_shoulder_elbow, lh_slope_value, rh_hip, lh_hip, lh_finger_wrist_elbow_coor,\
        # lh_knees, rh_knees, lh_shoulder, rh_shoulder, lh_ear, rh_ear = body_landmark_segment(frame.copy())
        skipFrameRes, ErhuLine, BowLine = erhu_bow_segment(frame.copy(), subtractor, skipFrame, horizontalLinePoint1,
                                                           horizontalLinePoint2, verticalLinePoint1, verticalLinePoint2,
                                                           horizontalAvgSize, verticalAvgSize, horizontalArray1,
                                                           horizontalArray2, verticalArray1, verticalArray2,
                                                           crossDistanceMaxH , crossDistanceMaxV, handPointsAvgSize,
                                                           rightHandPoints, leftHandPoints)
        skipFrame = skipFrameRes
        # cv2.imshow('RightHand', RH_Cropped)
        # cv2.imshow('RightArm', RArm_Cropped)
        # cv2.imshow('LeftArm', LArm_Cropped)
        # cv2.imshow('LeftHand', LH_Cropped)
        # cv2.imshow('Img Ori', image)
        # cv2.waitKey(1)

        if len(x_test_video_leftHand_resized) < limit_sample:
            # img_resized_leftHand = cv2.resize(LH_Cropped, (img_height, img_width))
            x_test_video_leftHand_resized.append(LH_Cropped)
            x_test_left_hand_point.append(LH_Cropped_Coor)
            x_test_left_hand_keypoint.append(leftHand_keypoints)
        if len(x_test_video_rightHand_resized) < limit_sample:
            # img_resized_rightHand = cv2.resize(RH_Cropped, (img_height, img_width))
            x_test_video_rightHand_resized.append(RH_Cropped)
            x_test_right_hand_point.append(RH_Cropped_Coor)
            x_test_right_hand_keypoint.append(rightHand_keypoints)
        if len(x_test_video_leftArm_resized) < limit_sample:
            # img_resized_leftArm = cv2.resize(LArm_Cropped, (img_height, img_width))
            x_test_video_leftArm_resized.append(LArm_Cropped)
            x_test_left_arm_point.append(LArm_Cropped_Coor)
            x_test_left_arm_keypoint.append(leftArm_keypoint)
        if len(x_test_video_rightArm_resized) < limit_sample:
            # img_resized_rightArm = cv2.resize(RArm_Cropped, (img_height, img_width))
            x_test_video_rightArm_resized.append(RArm_Cropped)
            x_test_right_arm_point.append(RArm_Cropped_Coor)
        if len(x_test_all_body_point) < limit_sample:
            x_test_all_body_point.append(
                [head_coordinate, body_coordinate, knees_shoulder_distance, degree_ear_face,
                 degree_body, degree_shoulder, lh_hip, rh_hip, degree_lh_shoulder_elbow, lh_slope_value,
                 lh_finger_wrist_elbow_coor, lh_knees, rh_knees, lh_shoulder, rh_shoulder, lh_ear, rh_ear])
        if len(x_test_erhu_line_point) < limit_sample:
            # erhu_line = [(350, 100), (350, 400)]
            # x_test_erhu_line_point.append(erhu_line)
            erhu_line = ErhuLine
            # print('Erhu Line:', erhu_line)
            x_test_erhu_line_point.append(erhu_line)
        if len(x_test_bow_line_point) < limit_sample:
            # bow_line = [(100, 300), (400, 300)]
            # x_test_bow_line_point.append(bow_line)
            bow_line = BowLine
            # print('Bow Line:', bow_line)
            x_test_bow_line_point.append(bow_line)

        # seg_mask, seg_output = segment_image.segmentFrame(frame.copy())
        # segLeng = len(seg_mask['scores'])
        # # print(seg_mask['scores'])
        # # print('Number of Segment:', segLeng)
        # # if segLeng >= 3:
        # # img_body = body_dilation(frame.copy())
        # image = frame.copy()
        # # print("Frame ", str(frame_number), " processed!")
        # for i in range(segLeng):
        #     mask = frame.copy()
        #     id = seg_mask['class_ids'][i]
        #     label = LABELS[int(id) - 1]
        #     if mask.shape[0] == seg_mask['masks'].shape[0] and mask.shape[1] == seg_mask['masks'].shape[1]:
        #         mask[seg_mask['masks'][:, :, i] == False] = (0, 0, 0)
        #         # videoSegmentation[label].write(mask)
        #         if label == 'erhu':
        #             # cv2.imshow('erhu', mask)
        #             # cv2.waitKey(1)
        #             # counter_erhu +=1
        #             try:
        #                 img, erhu_line = new_erhu_segment(mask)
        #             except:
        #                 img, erhu_line = new_erhu_segment(image)
        #             # image, erhu_line, QinZhenPart, QinThongPart = erhu_segment(image, mask)
        #             limit_counter_erhu += 1
        #             if len(x_test_erhu_line_point) < limit_sample:
        #                 x_test_erhu_line_point.append(erhu_line)
        #             # if len(x_test_video_qinThong_resized) < limit_sample:
        #             #     img_resized_qinThong = cv2.resize(QinThongPart, (img_height, img_width))
        #             #     x_test_video_qinThong_resized.append(img_resized_qinThong)
        #             #     x_test_video_qinThong.append(mask)
        #             # if len(x_test_video_qinZhen_resized) < limit_sample:
        #             #     img_resized_qinZhen = cv2.resize(QinZhenPart, (img_height, img_width))
        #             #     x_test_video_qinZhen_resized.append(img_resized_qinZhen)
        #             #     x_test_video_qinZhen.append(mask)
        #         elif label == 'body':
        #             # counter_body += 1
        #             # cv2.imshow('flip', image)
        #             # cv2.waitKey(1000)
        #             try:
        #                 image, rightHandPart, rightArmPart, leftHandPart, leftArmPart, \
        #                 halfBodyPart, leftArmPartOri, rightArmPartOri, leftHandPartOri, \
        #                 rightHandPartOri, left_arm_point, right_arm_point, right_hand_point, head_coordinate, body_coordinate, \
        #                 knees_shoulder_distance, degree_ear_face, degree_body, degree_shoulder, hip_left, hip_right = body_segment(
        #                     image, mask)
        #             except:
        #                 image, rightHandPart, rightArmPart, leftHandPart, leftArmPart, \
        #                 halfBodyPart, leftArmPartOri, rightArmPartOri, leftHandPartOri, \
        #                 rightHandPartOri, left_arm_point, right_arm_point, right_hand_point, head_coordinate, body_coordinate, \
        #                 knees_shoulder_distance, degree_ear_face, degree_body, degree_shoulder, hip_left, hip_right = body_segment(
        #                     image, image)
        #             limit_counter_body += 1
        #             # cv2.imshow('Mask', image)
        #             # cv2.imshow('leftHand', leftHandPart)
        #             # cv2.imshow('leftArm', leftArmPart)
        #             # cv2.imshow('RightArm', rightArmPart)
        #             # cv2.imshow('RightHand', rightHandPart)
        #             # cv2.waitKey(1)
        #             if len(x_test_video_leftHand_resized) < limit_sample:
        #                 img_resized_leftHand = cv2.resize(leftHandPart, (img_height, img_width))
        #                 x_test_video_leftHand_resized.append(img_resized_leftHand)
        #                 x_test_video_leftHand.append(leftHandPartOri)
        #             if len(x_test_video_rightHand_resized) < limit_sample:
        #                 img_resized_rightHand = cv2.resize(rightHandPart, (img_height, img_width))
        #                 x_test_video_rightHand_resized.append(img_resized_rightHand)
        #                 x_test_video_rightHand.append(rightHandPartOri)
        #                 x_test_right_hand_point.append(right_hand_point)
        #             if len(x_test_video_leftArm_resized) < limit_sample:
        #                 img_resized_leftArm = cv2.resize(leftArmPart, (img_height, img_width))
        #                 x_test_video_leftArm_resized.append(img_resized_leftArm)
        #                 x_test_video_leftArm.append(leftArmPartOri)
        #                 x_test_left_arm_point.append(left_arm_point)
        #             if len(x_test_video_rightArm_resized) < limit_sample:
        #                 img_resized_rightArm = cv2.resize(rightArmPart, (img_height, img_width))
        #                 x_test_video_rightArm_resized.append(img_resized_rightArm)
        #                 x_test_video_rightArm.append(rightArmPartOri)
        #                 x_test_right_arm_point.append(right_arm_point)
        #             if len(x_test_all_body_point) < limit_sample:
        #                 x_test_all_body_point.append(
        #                     [head_coordinate, body_coordinate, knees_shoulder_distance, degree_ear_face,
        #                      degree_body, degree_shoulder, hip_left, hip_right])
        #         elif label == 'bow':
        #             # cv2.imshow('bow', mask)
        #             # cv2.waitKey(1)
        #             # counter_bow += 1
        #             try:
        #                 img, bow_line = new_bow_segment(mask)
        #             except:
        #                 img, bow_line = new_bow_segment(image)
        #             # image, bow_line, bow_cropped, bow_angle = bow_segment(image, mask)
        #             limit_counter_bow += 1
        #             if len(x_test_bow_line_point) < limit_sample:
        #                 # img_resized_bow = cv2.resize(bow_cropped, (img_height, img_width))
        #                 # x_test_video_bow_resized.append(img_resized_bow)
        #                 x_test_bow_line_point.append(bow_line)
        #                 # x_test_video_bow.append(mask)
        #                 # x_test_bow_point.append(bow_angle)
        # # is_orthogonal = check_orthogonal(erhu_line[0][0], erhu_line[1][1], erhu_line[1][0], erhu_line[1][1], bow_line[0][0], bow_line[0][1], bow_line[1][0], bow_line[1][1])
        x_test_video_original.append(image)
        print('Read Frame -', frame_count, 'of', length)
        frame_number += 1
        limit_counter += 1
        frame_count += 1
        # except:
        #     print("Something is wrong...")
        # # else:
        # #     skipped_frame += 1
        # else:
        if limit_counter >= limit_sample:
            # exit()
            rightArm_E31_ClassName = '拇指握弓位置錯誤'
            rightArm_E32_ClassName = '食指握弓桿位置錯誤'
            rightArm_E33_ClassName = '中指或無名指觸弓位置錯誤'
            rightArm_E34_ClassName = '右手腕持弓太向內翻'
            rightArm_E35_ClassName = '右手腕持弓太向外翻'
            rightArm_Normal_ClassName = 'N-右手正常持弓 '
            # leftArm_E21 = round(np.argmax(prediction_leftArm),2)
            # leftArm_E22 = round(np.argmax(prediction_leftArm),2)
            # leftArm_E23 = round(np.argmax(prediction_leftHand),2)
            # print('leftArm_E21:', leftArm_E21)
            # print('leftArm_E22:', leftArm_E22)
            # print('leftArm_E23:', leftArm_E23)
            # print('rightArm_E31:', rightArm_E31)
            # print('rightArm_E32:', rightArm_E32)
            # print('rightArm_E33:', rightArm_E33)
            # print('rightArm_E34:', rightArm_E34)
            # print('rightArm_E35:', rightArm_E35)
            leftArm_Normal_ClassName = '左手持琴: 正常'
            leftArm_E21_ClassName = '左手持琴: 手臂過高'
            leftArm_E22_ClassName = '左手持琴: 手臂太低'
            leftArm_E23_ClassName = '左手持琴: 手腕手肘盡量一直線'
            E11_classname = '頭: 要擺正'
            E14_classname = '坐姿: 要坐正'
            E15_classname = '兩膝: 要與肩同寬'
            E13_classname = '肩: 右肩太高'
            E12_classname = '肩: 左肩太高'
            E41_classname = '琴桿: 左傾斜'
            E42_classname = '琴桿: 右傾斜'
            E43_classname = '運弓: 軌跡必須一直線'
            E44_classname = '運弓: 運弓太高'
            E45_classname = '運弓: 運弓太低'
            leftHand_AL1_ClassName = '左手按弦: 大拇指不可往上翹'
            leftHand_AL2_ClassName = '左手按弦: 手掌不可緊靠琴桿'
            err_face = 0
            err_body = 0
            err_orthogonal = 0
            err_bow = 0
            err_erhu_left = 0
            err_erhu_right = 0
            err_right_shoulder = 0
            err_left_shoulder = 0
            err_knees_shoulder = 0
            err_left_arm_high = 0
            err_left_arm_low = 0
            err_left_hand_slope = 0
            is_face_err = False
            is_body_err = False
            is_left_shoulder_err = False
            is_right_shoulder_err = False
            is_bow_err = False
            is_knees_shoulder = False
            is_erhu_left = False
            is_erhu_right = False
            is_not_orthogonal = False
            is_left_arm_too_high = False
            is_left_arm_too_low = False
            is_left_hand_slope = False
            is_bow_high = False
            is_bow_low = False
            is_right_hand_E31 = False
            is_right_hand_E32 = False
            is_right_hand_E33 = False
            is_right_hand_E3N = False
            prev_bow_angle = 0
            err_bow_high = 0
            err_bow_low = 0
            rightArm_E31 = 0
            rightArm_E32 = 0
            rightArm_E33 = 0
            rightArm_NHand = 0
            rightArm_E34 = 0
            rightArm_E35 = 0
            leftArm_E21 = 0
            leftArm_E22 = 0
            leftArm_E23 = 0
            leftArm_N = 0
            for (imgLeftHand, imgRightHand, imgLeftArm, imgRightArm, leftHandPoint, rightHandPoint, leftArmPoint, \
                rightArmPoint, all_body_point, leftHand_keypoint,\
                    leftArm_keypoint, rightHand_keypoint) in zip(x_test_video_leftHand_resized, x_test_video_rightHand_resized,
                                                     x_test_video_leftArm_resized, x_test_video_rightArm_resized,
                                                     x_test_left_hand_point, x_test_right_hand_point,x_test_left_arm_point,
                                                     x_test_right_arm_point, x_test_all_body_point, x_test_left_arm_keypoint,
                                                     x_test_left_hand_keypoint, x_test_right_hand_keypoint):
                limit_counter = 0
                # limit_counter_bow = 0
                # limit_counter_body = 0
                # limit_counter_erhu = 0
                # counter_leftHand = len(x_test_video_leftHand_resized)
                # counter_leftArm = len(x_test_video_leftArm_resized)
                # counter_rightHand = len(x_test_video_rightHand_resized)
                # counter_rightArm = len(x_test_video_rightArm_resized)
                # counter_bow_line = len(x_test_bow_line_point)
                # counter_erhu_line = len(x_test_erhu_line_point)
                # print(counter_leftArm, counter_rightHand, counter_rightArm, counter_leftHand)
                # # Set for list that less than limit sample and append with default value
                # canvas_padding = np.zeros(x_test_video_leftArm_resized[0].shape, np.uint8)
                # default_bow_line_coor = [(-1, -1), (-1, -1)]
                # default_erhu_line_coor = [(-1, -1), (-1, -1)]
                # if counter_leftHand < limit_sample:
                #     for i in range(limit_sample - counter_leftHand):
                #         x_test_video_leftHand_resized.append(canvas_padding)
                # if counter_leftArm < limit_sample:
                #     for i in range(limit_sample - counter_leftArm):
                #         x_test_video_leftArm_resized.append(canvas_padding)
                # if counter_rightArm < limit_sample:
                #     for i in range(limit_sample - counter_rightArm):
                #         x_test_video_rightArm_resized.append(canvas_padding)
                # if counter_rightHand < limit_sample:
                #     for i in range(limit_sample - counter_rightHand):
                #         x_test_video_rightHand_resized.append(canvas_padding)
                # if counter_bow_line < limit_sample:
                #     for i in range(limit_sample - counter_bow_line):
                #         print('Add bow line')
                #         x_test_bow_line_point.append(default_bow_line_coor)
                # if counter_erhu_line < limit_sample:
                #     for i in range(limit_sample - counter_erhu_line):
                #         x_test_erhu_line_point.append(default_erhu_line_coor)
                # print('Original Image Count:', len(x_test_video_original))
                # print('Image Bow Count:', len(x_test_video_bow_resized))
                # print('Image LeftArm Count:', len(x_test_video_leftArm_resized))
                # print('Image LeftHand Count:', len(x_test_video_leftHand_resized))
                # print('Image RightArm Count:', len(x_test_video_rightArm_resized))
                # print('Image RightHand Count:', len(x_test_video_rightHand_resized))
                # print('All body point Count:', len(x_test_all_body_point))
                # print('All bow line point Count:', len(x_test_bow_line_point))
                # print('All erhu line point Count:', len(x_test_erhu_line_point))
                # x_test_all_body_point, x_test_bow_line_point, x_test_erhu_line_point
                # print('Image QinThong Count:', len(x_test_video_qinThong_resized))
                # print('Image QinZhen Count:', len(x_test_video_qinZhen_resized))
                # x_test_bow.append(x_test_video_bow_resized)
                # x_test_leftArm.append(x_test_video_leftArm_resized)
                # x_test_leftHand.append(x_test_video_leftHand_resized)
                # x_test_rightArm.append(x_test_video_rightArm_resized)
                # x_test_rightHand.append(x_test_video_rightHand_resized)
                # x_test_qinThong.append(x_test_video_qinThong_resized)
                # x_test_qinZhen.append(x_test_video_qinZhen_resized)
                # # test_x_bow          = np.array(x_test_bow)
                # # print('bow shape:', test_x_bow.shape)
                # # test_x_bow          = test_x_bow / 255.0
                # # print(len(x_test_rightArm))
                # # test_x_leftArm = np.array(x_test_leftArm)
                # # # print('leftArm shape:', test_x_leftArm.shape)
                # # test_x_leftArm = test_x_leftArm / 255.0
                #
                # # test_x_leftHand = np.array(x_test_leftHand)
                # # # print('leftHand shape:', test_x_leftHand.shape)
                # # test_x_leftHand = test_x_leftHand / 255.0
                # print(len(x_test_rightArm))
                # test_x_rightArm = np.array(x_test_rightArm)
                # test_x_rightArm = test_x_rightArm / 255.0
                # test_x_rightHand = np.array(x_test_rightHand)
                # test_x_rightHand = test_x_rightHand / 255.0
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
                # prediction_leftArm = model_left_arm.predict(test_x_leftArm)
                # prediction_leftHand = model_left.predict(test_x_leftHand)
                # prediction_rightArm = model.predict(test_x_rightArm)
                x_train = []
                y_train = []
                edge_train = []
                edge_train.append(hand_edge2)
                x_train.append(rightHand_keypoint)
                h, w, c = imgRightHand.shape
                # img_padding = img_paddings
                scale_w = 416 / w
                color = [255, 255, 255]
                h_scaled = int(h * scale_w)
                img_scaled = cv2.resize(imgRightHand, (img_height, h_scaled))
                # cv2.imshow('imgRightHand', imgRightHand)
                # cv2.imshow('img_scaled', img_scaled)
                # cv2.waitKey(1)
                try:
                    img_padding = cv2.copyMakeBorder(img_scaled, 0, (img_height - h_scaled), 0, 0, cv2.BORDER_CONSTANT, value=color)
                except:
                    img_padding = cv2.resize(img_scaled, (416, 416))
                x_train_yolo = []
                x_train_yolo.append(img_padding)
                yolo_x = np.asarray(x_train_yolo)
                train_x = np.asarray(x_train)
                edge = np.asarray(edge_train)
                train_x = NormalizeData(train_x)
                prediction_rightHand = model.predict([yolo_x, train_x, edge])

                # print(prediction_rightArm)
                # print(prediction_rightHand[0])
                # print(prediction_leftArm)
                # print(prediction_leftHand)
                # prediction_right_arm_max = np.argmax(prediction_rightArm[0])
                # prediction_left_max = np.argmax(prediction_leftArm[0])
                prediction_right_hand_max = np.argmax(prediction_rightHand[0])
                # print(prediction_right_hand_max)


                if prediction_right_hand_max == 0:
                    if (round((prediction_rightHand[0][0]), 2) > P_var):
                        rightArm_E31 += 1
                    # rightArm_E31 = round((prediction_rightHand[0][0]), 2)
                elif prediction_right_hand_max == 1:
                    if (round((prediction_rightHand[0][1]), 2) > P_var):
                        rightArm_E32 += 1
                    # rightArm_E32 = round((prediction_rightHand[0][1]), 2)
                elif prediction_right_hand_max == 2:
                    if (round((prediction_rightHand[0][2]), 2) > P_var):
                        rightArm_E33 += 1
                    # rightArm_E33 = round((prediction_rightHand[0][2]), 2)
                elif prediction_right_hand_max == 3:
                    if (round((prediction_rightHand[0][3]), 2) > P_var):
                        rightArm_NHand += 1
                    # rightArm_NHand = round((prediction_rightHand[0][3]), 2)
                # if prediction_right_arm_max == 0:
                #     rightArm_E34 = round((prediction_rightArm[0][0]), 2)
                # elif prediction_right_arm_max == 1:
                #     rightArm_E35 = round((prediction_rightArm[0][1]), 2)
                # elif prediction_right_hand_max == 2:
                #     rightArm_NArm = round((prediction_rightArm[0][2]), 2)

                # if prediction_left_max == 0:
                #     leftArm_E21 = round((prediction_leftArm[0][0]), 2)
                # elif prediction_left_max == 1:
                #     leftArm_E22 = round((prediction_leftArm[0][1]), 2)
                # elif prediction_left_max == 2:
                #     leftArm_E23 = round((prediction_leftArm[0][2]), 2)
                # elif prediction_left_max == 3:
                #     leftArm_N = round((prediction_leftArm[0][3]), 2)

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

                # for (all_body_point, bow_line, erhu_line) in zip(x_test_all_body_point, x_test_bow_line_point, x_test_erhu_line_point):
                head_coordinate = all_body_point[0]
                body_coordinate = all_body_point[1]
                knees_shoulder_distance = all_body_point[2]
                degree_ear_face = all_body_point[3]
                degree_body = all_body_point[4]
                degree_shoulder = all_body_point[5]
                hip_left_point = all_body_point[6]
                hip_right_point = all_body_point[7]
                degree_lh_shoulder_elbow = all_body_point[8]
                lh_slope_value = all_body_point[9]
                lh_finger_wrist_elbow_coor = all_body_point[10]
                lh_knees_coor = all_body_point[11]
                rh_knees_coor = all_body_point[12]
                lh_shoulder_coor = all_body_point[13]
                rh_shoulder_coor = all_body_point[14]
                lh_ear_coor = all_body_point[15]
                rh_ear_coor = all_body_point[16]

                if degree_ear_face >= K_var or degree_ear_face < (0 - K_var):
                    err_face += 1
                if degree_body > (90 + K_var) or degree_body < (90 - K_var):
                    err_body += 1
                if degree_shoulder > ((X_var / 90) * 100):
                    err_right_shoulder += 1
                elif degree_shoulder < (0 - ((X_var / 90) * 100)):
                    err_left_shoulder += 1
                if abs(knees_shoulder_distance) > KN_var:
                    err_knees_shoulder += 1
                if degree_lh_shoulder_elbow < K_LArm_high_var:
                    err_left_arm_high += 1
                elif degree_lh_shoulder_elbow > K_LArm_low_var:
                    err_left_arm_low += 1
                if lh_slope_value > S_var:
                    err_left_hand_slope += 1

                L1_angle = get_angle(erhu_line[0], erhu_line[1])
                L2_angle = get_angle(bow_line[0], bow_line[1])
                orthogonal_angle = abs(abs(L1_angle) - abs(L2_angle))
                if (orthogonal_angle <= 90 - N_var or orthogonal_angle >= 90 + N_var) and erhu_line[0][0]!=0 and erhu_line[0][1]!=0 and erhu_line[1][0]!=0 and erhu_line[1][1]!=0:
                    err_orthogonal += 1
                    if L1_angle > (90 + N_var):
                        err_erhu_right += 1
                    else:
                        err_erhu_left += 1
                # if (abs(prev_bow_angle-L2_angle) > (0 + N_var) or abs(prev_bow_angle-L2_angle) < (0 - N_var)):
                #     err_bow += 1
                if L2_angle > 9 :
                    err_bow_high += 1
                    # print(L2_angle)
                elif L2_angle < -9:
                    err_bow_low += 1
                    # print(L2_angle)
                elif abs(prev_bow_angle-L2_angle) > M_var:
                    err_bow += 1
                prev_bow_angle = L2_angle

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
            if err_left_arm_high > limit_sample // 2:
                is_left_arm_too_high = True
            if err_left_arm_low > limit_sample // 2:
                is_left_arm_too_low = True
            if err_left_hand_slope > limit_sample // 2:
                is_left_hand_slope = True
            if err_bow_high > limit_sample // 2:
                is_bow_high = True
            if err_bow_low > limit_sample // 2:
                is_bow_low = True
            if rightArm_E31 > limit_sample // 3:
                is_right_hand_E31 = True
            elif rightArm_E32 > limit_sample // 3:
                is_right_hand_E32 = True
            elif rightArm_E33 > limit_sample // 3:
                is_right_hand_E33 = True
            elif rightArm_NHand > limit_sample // 3:
                is_right_hand_E3N = True
            else:
                is_right_hand_E3N = True
            print('rightArm_E31:', rightArm_E31)
            print('rightArm_E32:', rightArm_E32)
            print('rightArm_E33:', rightArm_E33)
            print('rightArm_E3N:', rightArm_NHand)

            for img_original, ori_right_hand_point, ori_right_arm_point, ori_left_arm_point, ori_left_hand_point, all_body_point, bow_line, erhu_line in \
                    zip(x_test_video_original, x_test_right_hand_point, x_test_right_arm_point, x_test_left_arm_point, x_test_left_hand_point,
                        x_test_all_body_point, x_test_bow_line_point, x_test_erhu_line_point):
                idx_write_frame += 1
                warning_mess = []
                img_pil = Image.fromarray(img_original)
                img_err = Image.fromarray(img_original)
                draw_res_bow = ImageDraw.Draw(img_pil)
                draw_res_err_box = ImageDraw.Draw(img_err)
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
                degree_lh_shoulder_elbow = all_body_point[8]
                lh_slope_value = all_body_point[9]
                lh_finger_wrist_elbow_coor = all_body_point[10]
                lh_knees_coor = all_body_point[11]
                rh_knees_coor = all_body_point[12]
                lh_shoulder_coor = all_body_point[13]
                rh_shoulder_coor = all_body_point[14]
                lh_ear_coor = all_body_point[15]
                rh_ear_coor = all_body_point[16]

                # cv2.rectangle(img_chinese, (10, 10), (200, 200), (255, 255, 255))
                # draw_res_bow.rectangle([(5, 5), (500, 350)], outline=None, fill="#ffffff")
                div_space_val = 10
                err_mess_coor = [1, 1]
                draw_res_err_box.rectangle(
                    [(err_mess_coor[0], err_mess_coor[1]), (err_mess_frame[0] * 1.5, err_mess_frame[1])], outline=None,
                    fill=(200, 200, 200))
                # cv2.rectangle(img_chinese, (10, 10), (100, 100), (255, 255, 255))
                # draw_res_bow.rectangle([(5, 5), (300, 150)], outline=None, fill="#ffffff")
                # draw_res_bow.text((10, 20), "===== Traditional Algorithm =====", font=result_font, fill=(b, g, r, a))

                # E11 : Head position not normal ===================================================================== 0
                # if degree_ear_face >= K_var or degree_ear_face < (0-K_var):
                head_err_mess_coor = [err_mess_coor[0] + int(err_mess_frame[0] // 20),
                                      err_mess_coor[1] + int(err_mess_frame[0] // 20)]
                if is_face_err == True:
                    # print('Face')
                    # draw_res_bow.text((10, 50), E11_classname + ':' + str(degree_ear_face), font=result_font, fill=(b, g, r, a))
                    draw_res_err_box.text((head_err_mess_coor[0], head_err_mess_coor[1]), E11_classname,
                                          font=result_font, fill=(0, 128, 255))
                    warning_mess.append(["E11", str(degree_ear_face), 'Head Position', str(degree_ear_face),
                                         'E11-Head position not normal (to L or R)'])
                    # draw_res_bow.rectangle(head_coordinate, outline="blue", fill=None, width=4)
                    r_circle = 5
                    draw_res_bow.ellipse(
                        [(lh_ear_coor[0] - r_circle, lh_ear_coor[1] - r_circle),
                         (lh_ear_coor[0] + r_circle, lh_ear_coor[1] + r_circle)],
                        fill=(0, 128, 255))
                    draw_res_bow.ellipse(
                        [(rh_ear_coor[0] - r_circle, rh_ear_coor[1] - r_circle),
                         (rh_ear_coor[0] + r_circle, rh_ear_coor[1] + r_circle)],
                        fill=(0, 128, 255))
                    draw_res_err_box.ellipse(
                        [(lh_ear_coor[0] - r_circle, lh_ear_coor[1] - r_circle),
                         (lh_ear_coor[0] + r_circle, lh_ear_coor[1] + r_circle)],
                        fill=(0, 128, 255))
                    draw_res_err_box.ellipse(
                        [(rh_ear_coor[0] - r_circle, rh_ear_coor[1] - r_circle),
                         (rh_ear_coor[0] + r_circle, rh_ear_coor[1] + r_circle)],
                        fill=(0, 128, 255))
                else:
                    # print('Else Face')
                    # draw_res_bow.text((10, 50), 'Head Position : Normal', font=result_font, fill=(b, g, r, a))
                    draw_res_err_box.text((head_err_mess_coor[0], head_err_mess_coor[1]), '頭: 正常', font=result_font,
                                          fill=(0, 0, 0))
                    warning_mess.append(["Head_Normal", "Normal", 'Head Position', str(0), 'Normal'])

                # E14 : Error Body/Sitting Position ==================================================================== 1
                # if degree_body > (90+K_var) or degree_body < (90-K_var):
                body_err_mess_coor = [head_err_mess_coor[0],
                                      head_err_mess_coor[1] + int(err_mess_frame[0] // div_space_val)]
                if is_body_err == True:
                    # print('Body')
                    # draw_res_bow.text((10, 80), E14_classname + ':' + str(degree_body), font=result_font, fill=(b, g, r, a))
                    draw_res_err_box.text((body_err_mess_coor[0], body_err_mess_coor[1]), E14_classname,
                                          font=result_font, fill=(0, 128, 255))
                    warning_mess.append(["E14", str(degree_body), 'Body Position', str(degree_body),
                                         'E14-Need to seat straight (to L or R)'])
                    draw_res_bow.rectangle(body_coordinate, outline=(0, 128, 255), fill=None, width=4)
                    draw_res_err_box.rectangle(body_coordinate, outline=(0, 128, 255), fill=None, width=4)
                else:
                    # print('Else Body')
                    # draw_res_bow.text((10, 80), 'Body Position : Normal', font=result_font, fill=(b, g, r, a))
                    draw_res_err_box.text((body_err_mess_coor[0], body_err_mess_coor[1]), '坐姿: 正常', font=result_font,
                                          fill=(0, 0, 0))
                    warning_mess.append(["Body_Normal", "Normal", 'Body Position', str(0), 'Normal'])

                # E12 - E13 Left Shoulder too High / Right Shoulder too High ========================================= 2
                # if degree_shoulder > ((X_var/90)*100):
                shoulder_err_mess_coor = [body_err_mess_coor[0],
                                          body_err_mess_coor[1] + int(err_mess_frame[0] // div_space_val)]
                if is_right_shoulder_err == True:
                    # print('Shoulder')
                    # draw_res_bow.text((10, 140), E13_classname + ':' + str(degree_shoulder), font=result_font, fill=(b, g, r, a))
                    draw_res_err_box.text((shoulder_err_mess_coor[0], shoulder_err_mess_coor[1]), E13_classname,
                                          font=result_font, fill=(0, 128, 255))
                    warning_mess.append(["E13", str(degree_shoulder), 'Shoulder Position', str(degree_shoulder),
                                         'E13-Right shoulder too hight '])
                    # draw_res_bow.rectangle(body_coordinate, outline="blue", fill=None, width=4)
                    r_circle = 7
                    draw_res_bow.ellipse(
                        [(rh_shoulder_coor[0] - r_circle, rh_shoulder_coor[1] - r_circle),
                         (rh_shoulder_coor[0] + r_circle, rh_shoulder_coor[1] + r_circle)],
                        fill=(0, 128, 255))
                    draw_res_err_box.ellipse(
                        [(rh_shoulder_coor[0] - r_circle, rh_shoulder_coor[1] - r_circle),
                         (rh_shoulder_coor[0] + r_circle, rh_shoulder_coor[1] + r_circle)],
                        fill=(0, 128, 255))
                # elif degree_shoulder < (0-((X_var/90)*100)):
                elif is_left_shoulder_err == True:
                    # print('Else Shoulder')
                    # draw_res_bow.text((10, 140), E12_classname + ':' + str(degree_shoulder), font=result_font, fill=(b, g, r, a))
                    draw_res_err_box.text((shoulder_err_mess_coor[0], shoulder_err_mess_coor[1]), E12_classname,
                                          font=result_font, fill=(0, 128, 255))
                    draw_res_err_box.text((shoulder_err_mess_coor[0], shoulder_err_mess_coor[1]), E12_classname,
                                          font=result_font, fill=(0, 128, 255))
                    warning_mess.append(["E12", str(degree_shoulder), 'Shoulder Position', str(degree_shoulder),
                                         'E12-Left shoulder too hight'])
                    # draw_res_bow.rectangle(body_coordinate, outline="blue", fill=None, width=4)
                    r_circle = 7
                    draw_res_bow.ellipse(
                        [(lh_shoulder_coor[0] - r_circle, lh_shoulder_coor[1] - r_circle),
                         (lh_shoulder_coor[0] + r_circle, lh_shoulder_coor[1] + r_circle)],
                        fill=(0, 128, 255))
                    draw_res_err_box.ellipse(
                        [(lh_shoulder_coor[0] - r_circle, lh_shoulder_coor[1] - r_circle),
                         (lh_shoulder_coor[0] + r_circle, lh_shoulder_coor[1] + r_circle)],
                        fill=(0, 128, 255))
                else:
                    # print('Else Shoulder')
                    # draw_res_bow.text((10, 140), 'Shoulders Position : Normal', font=result_font, fill=(b, g, r, a))
                    draw_res_err_box.text((shoulder_err_mess_coor[0], shoulder_err_mess_coor[1]), '肩: 正常',
                                          font=result_font, fill=(0, 0, 0))
                    warning_mess.append(['Shoulder_Normal', 'Normal', 'Shoulders Position', str(0), 'Normal'])

                # E41/E42 Erhu Position ==================================================================================== 5
                erhu_err_mess_coor = [shoulder_err_mess_coor[0],
                                      shoulder_err_mess_coor[1] + int(err_mess_frame[0] // div_space_val)]
                erhu_line_shape = erhu_line
                bow_line_shape = bow_line
                # print('Erhu Bow Line Shape:', bow_line_shape, erhu_line_shape)
                # if is_orthogonal == False:
                if is_bow_high == False and is_bow_low == False and is_bow_err == False:
                    if is_not_orthogonal == True:
                        L1_angle = get_angle(erhu_line[0], erhu_line[1])
                        L2_angle = get_angle(bow_line[0], bow_line[1])
                        if is_erhu_left == True:
                            # draw_res_bow.text((10, 320), E41_classname, font=result_font, fill=(b, g, r, a))
                            draw_res_err_box.text((erhu_err_mess_coor[0], erhu_err_mess_coor[1]), E41_classname,
                                                  font=result_font, fill=(0, 255, 255))
                            warning_mess.append(["E41", str(L1_angle), "Erhu Position", str(L1_angle),
                                                 "E41-Pole tilt to left - Bow hair and string must be orthogonal"])
                            draw_res_bow.line(bow_line_shape, fill=(0, 255, 0), width=4)
                            draw_res_bow.line(erhu_line_shape, fill=(0, 255, 255), width=4)
                            draw_res_err_box.line(bow_line_shape, fill=(0, 255, 0), width=4)
                            draw_res_err_box.line(erhu_line_shape, fill=(0, 255, 255), width=4)
                        else:
                            # draw_res_bow.text((10, 320), E42_classname, font=result_font, fill=(b, g, r, a))
                            draw_res_err_box.text((erhu_err_mess_coor[0], erhu_err_mess_coor[1]), E42_classname,
                                                  font=result_font, fill=(0, 255, 255))
                            warning_mess.append(["E42", str(L1_angle), "Erhu Position", str(L1_angle),
                                                 "E42-Pole tilt to right - Bow hair and string must be orthogonal"])
                            draw_res_bow.line(bow_line_shape, fill=(0, 255, 0), width=4)
                            draw_res_bow.line(erhu_line_shape, fill=(0, 255, 255), width=4)
                            draw_res_err_box.line(bow_line_shape, fill=(0, 255, 0), width=4)
                            draw_res_err_box.line(erhu_line_shape, fill=(0, 255, 255), width=4)
                    else:
                        draw_res_err_box.text((erhu_err_mess_coor[0], erhu_err_mess_coor[1]), '琴桿: 正常', font=result_font,
                                              fill=(0, 0, 0))
                        warning_mess.append(["Erhu_Normal", "Normal", "Erhu Position", str(1.0), "Normal"])
                else:
                    draw_res_err_box.text((erhu_err_mess_coor[0], erhu_err_mess_coor[1]), '琴桿: 正常', font=result_font,
                                          fill=(0, 0, 0))
                    warning_mess.append(["Erhu_Normal", "Normal", "Erhu Position", str(1.0), "Normal"])

                # E43 Bow Trajectory must stright line ================================================================= 6
                bow_err_mess_coor = [erhu_err_mess_coor[0],
                                     erhu_err_mess_coor[1] + int(err_mess_frame[0] // div_space_val)]
                if is_bow_high == True:
                    draw_res_err_box.text((bow_err_mess_coor[0], bow_err_mess_coor[1]), E44_classname, font=result_font,
                                          fill=(0, 255, 0))
                    warning_mess.append(["E44", str(L2_angle), "Bow Position", str(L2_angle),
                                         "E44-Trace of bow too high"])
                    draw_res_bow.line(bow_line_shape, fill=(0, 255, 0), width=4)
                    draw_res_err_box.line(bow_line_shape, fill=(0, 255, 0), width=4)
                elif is_bow_low == True:
                    draw_res_err_box.text((bow_err_mess_coor[0], bow_err_mess_coor[1]), E45_classname, font=result_font,
                                          fill=(0, 255, 0))
                    warning_mess.append(["E45", str(L2_angle), "Bow Position", str(L2_angle),
                                         "E45-Trace of bow too high"])
                    draw_res_bow.line(bow_line_shape, fill=(0, 255, 0), width=4)
                    draw_res_err_box.line(bow_line_shape, fill=(0, 255, 0), width=4)
                elif is_bow_err == True:
                    # draw_res_bow.text((10, 320), E43_classname, font=result_font, fill=(b, g, r, a))
                    draw_res_err_box.text((bow_err_mess_coor[0], bow_err_mess_coor[1]), E43_classname, font=result_font,
                                          fill=(0, 255, 0))
                    warning_mess.append(["E43", str(L2_angle), "Bow Position", str(L2_angle),
                                         "E43-Trace of bow must be in straight line"])
                    draw_res_bow.line(bow_line_shape, fill=(0, 255, 0), width=4)
                    draw_res_err_box.line(bow_line_shape, fill=(0, 255, 0), width=4)
                    # draw_res_bow.line(erhu_line_shape, fill='blue', width=4)
                else:
                    # print('Else Orthogonal')
                    # draw_res_bow.text((10, 320), 'Bow and Erhu is Orthogonal', font=result_font, fill=(b, g, r, a))
                    draw_res_err_box.text((bow_err_mess_coor[0], bow_err_mess_coor[1]), '運弓: 正常', font=result_font,
                                          fill=(0, 0, 0))
                    warning_mess.append(["Bow_Normal", "Normal", "Bow Erhu Position", str(1.0), "Normal"])

                # E21/E22/E23 Error Left Arm ================================================================================= 4
                leftarm_err_mess_coor = [bow_err_mess_coor[0],
                                         bow_err_mess_coor[1] + int(err_mess_frame[0] // div_space_val)]
                if is_left_arm_too_high == True:
                    draw_res_err_box.text((leftarm_err_mess_coor[0], leftarm_err_mess_coor[1]),
                                          leftArm_E21_ClassName, font=result_font,
                                          fill=(0, 0, 255))
                    warning_mess.append(["E21", str(degree_lh_shoulder_elbow), 'Left Hand Arm Position',
                                         str(degree_lh_shoulder_elbow),
                                         'E21-Left elbow too Hight'])
                    # draw_res_bow.rectangle(ori_left_arm_rectangle_shape, outline='blue', fill=None, width=4)
                    try:
                        r_circle = 4
                        draw_res_bow.ellipse(
                            [(lh_shoulder_coor[0] - r_circle, lh_shoulder_coor[1] - r_circle),
                             (lh_shoulder_coor[0] + r_circle, lh_shoulder_coor[1] + r_circle)],
                            fill='blue')
                        draw_res_bow.line([(lh_shoulder_coor[0], lh_shoulder_coor[1]),
                                           (lh_finger_wrist_elbow_coor[3][0], lh_finger_wrist_elbow_coor[3][1])],
                                          fill='blue', width=4)
                        draw_res_err_box.ellipse([(lh_shoulder_coor[0] - r_circle, lh_shoulder_coor[1] - r_circle),
                                                  (lh_shoulder_coor[0] + r_circle, lh_shoulder_coor[1] + r_circle)],
                                                 fill='blue')
                        draw_res_err_box.line([(lh_shoulder_coor[0], lh_shoulder_coor[1]),
                                               (lh_finger_wrist_elbow_coor[3][0], lh_finger_wrist_elbow_coor[3][1])],
                                              fill='blue', width=4)
                    except:
                        print('Missing Mediapipe')
                # elif leftArm_E22 >= P_var:
                elif is_left_arm_too_low == True:
                    draw_res_err_box.text((leftarm_err_mess_coor[0], leftarm_err_mess_coor[1]),
                                          leftArm_E22_ClassName, font=result_font,
                                          fill=(0, 0, 255))
                    warning_mess.append(["E22", str(degree_lh_shoulder_elbow), 'Left Hand Arm Position',
                                         str(degree_lh_shoulder_elbow), 'E22-Left elbow too Low'])
                    # draw_res_bow.rectangle(ori_left_arm_rectangle_shape, outline='blue', fill=None, width=4)
                    try:
                        r_circle = 4
                        draw_res_bow.ellipse(
                            [(lh_shoulder_coor[0] - r_circle, lh_shoulder_coor[1] - r_circle),
                             (lh_shoulder_coor[0] + r_circle, lh_shoulder_coor[1] + r_circle)],
                            fill='blue')
                        draw_res_bow.line([(lh_shoulder_coor[0], lh_shoulder_coor[1]),
                                           (lh_finger_wrist_elbow_coor[3][0], lh_finger_wrist_elbow_coor[3][1])],
                                          fill='blue', width=4)
                        draw_res_err_box.ellipse([(lh_shoulder_coor[0] - r_circle, lh_shoulder_coor[1] - r_circle),
                                                  (lh_shoulder_coor[0] + r_circle, lh_shoulder_coor[1] + r_circle)],
                                                 fill='blue')
                        draw_res_err_box.line([(lh_shoulder_coor[0], lh_shoulder_coor[1]),
                                               (lh_finger_wrist_elbow_coor[3][0], lh_finger_wrist_elbow_coor[3][1])],
                                              fill='blue', width=4)
                    except:
                        print('Missing Mediapipe')
                elif is_left_hand_slope == True:
                    draw_res_err_box.text((leftarm_err_mess_coor[0], leftarm_err_mess_coor[1]),
                                          leftArm_E23_ClassName, font=result_font,
                                          fill=(0, 0, 255))
                    warning_mess.append(
                        ["E23", str(lh_slope_value), 'Left Hand Wrist Position', str(lh_slope_value),
                         'E23-Left elbow and wrist in a line'])
                    # draw_res_bow.rectangle(ori_left_hand_rectangle_shape, outline='blue', fill=None, width=4)
                    # print((lh_finger_wrist_elbow_coor[0][0], lh_finger_wrist_elbow_coor[0][1]), (lh_finger_wrist_elbow_coor[1][0], lh_finger_wrist_elbow_coor[1][1]), (lh_finger_wrist_elbow_coor[2][0], lh_finger_wrist_elbow_coor[2][1]))
                    try:
                        r_circle = 4
                        draw_res_bow.ellipse(
                            [(lh_finger_wrist_elbow_coor[0][0] - r_circle,
                              lh_finger_wrist_elbow_coor[0][1] - r_circle),
                             (
                                 lh_finger_wrist_elbow_coor[0][0] + r_circle,
                                 lh_finger_wrist_elbow_coor[0][1] + r_circle)],
                            fill='blue')
                        draw_res_bow.ellipse(
                            [(lh_finger_wrist_elbow_coor[1][0] - r_circle,
                              lh_finger_wrist_elbow_coor[1][1] - r_circle),
                             (
                                 lh_finger_wrist_elbow_coor[1][0] + r_circle,
                                 lh_finger_wrist_elbow_coor[1][1] + r_circle)],
                            fill='blue')
                        draw_res_bow.ellipse(
                            [(lh_finger_wrist_elbow_coor[2][0] - r_circle,
                              lh_finger_wrist_elbow_coor[2][1] - r_circle),
                             (
                                 lh_finger_wrist_elbow_coor[2][0] + r_circle,
                                 lh_finger_wrist_elbow_coor[2][1] + r_circle)],
                            fill='blue')
                        draw_res_bow.ellipse(
                            [(lh_finger_wrist_elbow_coor[3][0] - r_circle,
                              lh_finger_wrist_elbow_coor[3][1] - r_circle),
                             (
                                 lh_finger_wrist_elbow_coor[3][0] + r_circle,
                                 lh_finger_wrist_elbow_coor[3][1] + r_circle)],
                            fill='blue')
                        draw_res_bow.line([(lh_finger_wrist_elbow_coor[0][0], lh_finger_wrist_elbow_coor[0][1]),
                                           (lh_finger_wrist_elbow_coor[1][0], lh_finger_wrist_elbow_coor[1][1])],
                                          fill='blue', width=4)
                        draw_res_bow.line([(lh_finger_wrist_elbow_coor[1][0], lh_finger_wrist_elbow_coor[1][1]),
                                           (lh_finger_wrist_elbow_coor[2][0], lh_finger_wrist_elbow_coor[2][1])],
                                          fill='blue', width=4)
                        draw_res_bow.line([(lh_finger_wrist_elbow_coor[2][0], lh_finger_wrist_elbow_coor[2][1]),
                                           (lh_finger_wrist_elbow_coor[3][0], lh_finger_wrist_elbow_coor[3][1])],
                                          fill='blue', width=4)
                        draw_res_err_box.ellipse(
                            [(lh_finger_wrist_elbow_coor[0][0] - r_circle,
                              lh_finger_wrist_elbow_coor[0][1] - r_circle),
                             (
                                 lh_finger_wrist_elbow_coor[0][0] + r_circle,
                                 lh_finger_wrist_elbow_coor[0][1] + r_circle)],
                            fill='blue')
                        draw_res_err_box.ellipse(
                            [(lh_finger_wrist_elbow_coor[1][0] - r_circle,
                              lh_finger_wrist_elbow_coor[1][1] - r_circle),
                             (
                                 lh_finger_wrist_elbow_coor[1][0] + r_circle,
                                 lh_finger_wrist_elbow_coor[1][1] + r_circle)],
                            fill='blue')
                        draw_res_err_box.ellipse(
                            [(lh_finger_wrist_elbow_coor[2][0] - r_circle,
                              lh_finger_wrist_elbow_coor[2][1] - r_circle),
                             (
                                 lh_finger_wrist_elbow_coor[2][0] + r_circle,
                                 lh_finger_wrist_elbow_coor[2][1] + r_circle)],
                            fill='blue')
                        draw_res_err_box.ellipse(
                            [(lh_finger_wrist_elbow_coor[3][0] - r_circle,
                              lh_finger_wrist_elbow_coor[3][1] - r_circle),
                             (
                                 lh_finger_wrist_elbow_coor[3][0] + r_circle,
                                 lh_finger_wrist_elbow_coor[3][1] + r_circle)],
                            fill='blue')
                        draw_res_err_box.line([(lh_finger_wrist_elbow_coor[0][0], lh_finger_wrist_elbow_coor[0][1]),
                                               (lh_finger_wrist_elbow_coor[1][0], lh_finger_wrist_elbow_coor[1][1])],
                                              fill='blue', width=4)
                        draw_res_err_box.line([(lh_finger_wrist_elbow_coor[1][0], lh_finger_wrist_elbow_coor[1][1]),
                                               (lh_finger_wrist_elbow_coor[2][0], lh_finger_wrist_elbow_coor[2][1])],
                                              fill='blue', width=4)
                        draw_res_err_box.line([(lh_finger_wrist_elbow_coor[2][0], lh_finger_wrist_elbow_coor[2][1]),
                                               (lh_finger_wrist_elbow_coor[3][0], lh_finger_wrist_elbow_coor[3][1])],
                                              fill='blue', width=4)
                    except:
                        print('Body Landmark Missing')
                elif leftArm_N >= P_var:
                    draw_res_err_box.text((leftarm_err_mess_coor[0], leftarm_err_mess_coor[1]),
                                          leftArm_Normal_ClassName, font=result_font, fill=(0, 0, 0))
                    warning_mess.append(
                        ["LeftHand_Normal", "Normal", 'Left Hand Arm is Inline', str(1.0), 'Normal'])
                else:
                    draw_res_err_box.text((leftarm_err_mess_coor[0], leftarm_err_mess_coor[1]),
                                          leftArm_Normal_ClassName, font=result_font, fill=(0, 0, 0))
                    warning_mess.append(
                        ["LeftHand_Normal", "Normal", 'Left Hand Arm is Inline', str(1.0), 'Normal'])

                # E31 Error Right Hand ================================================================================ 3
                # draw_res_bow.text((10, 170), "===== Right Arm =====", font=result_font, fill=(b, g, r, a))
                rightarm_err_mess_coor = [leftarm_err_mess_coor[0],
                                          leftarm_err_mess_coor[1] + int(err_mess_frame[0] // div_space_val)]
                # if rightArm_E31 >= P_var:
                if is_right_hand_E31 == True:
                    draw_res_err_box.text((rightarm_err_mess_coor[0], rightarm_err_mess_coor[1]),
                                          rightArm_E31_ClassName, font=result_font, fill=(0, 0, 255))
                    warning_mess.append(["E31", str(rightArm_E31), 'Right Hand Hand Position', str(rightArm_E31),
                                         'E31-Wrong RH thumb position'])
                    draw_res_bow.rectangle(ori_right_hand_rectangle_shape, outline="blue", fill=None, width=4)
                    draw_res_err_box.rectangle(ori_right_hand_rectangle_shape, outline="blue", fill=None, width=4)
                # elif rightArm_E32 >= P_var:
                elif is_right_hand_E32 == True:
                    draw_res_err_box.text((rightarm_err_mess_coor[0], rightarm_err_mess_coor[1]),
                                          rightArm_E32_ClassName, font=result_font, fill=(0, 0, 255))
                    warning_mess.append(["E32", str(rightArm_E32), 'Right Hand Hand Position', str(rightArm_E32),
                                         'E32-Wrong RH index finger position'])
                    draw_res_bow.rectangle(ori_right_hand_rectangle_shape, outline="blue", fill=None, width=4)
                    draw_res_err_box.rectangle(ori_right_hand_rectangle_shape, outline="blue", fill=None, width=4)
                # elif rightArm_E33 >= P_var:
                elif is_right_hand_E33 == True:
                    draw_res_err_box.text((rightarm_err_mess_coor[0], rightarm_err_mess_coor[1]),
                                          rightArm_E33_ClassName, font=result_font, fill=(0, 0, 255))
                    warning_mess.append(["E33", str(rightArm_E33), 'Right Hand Hand Position', str(rightArm_E33),
                                         'E33-Wrong RH middle or ring finger position'])
                    draw_res_bow.rectangle(ori_right_hand_rectangle_shape, outline="blue", fill=None, width=4)
                    draw_res_err_box.rectangle(ori_right_hand_rectangle_shape, outline="blue", fill=None, width=4)
                # elif rightArm_NHand >= P_var:
                elif is_right_hand_E3N == True:
                    draw_res_err_box.text((rightarm_err_mess_coor[0], rightarm_err_mess_coor[1]), '右手持弓: 正常',
                                          font=result_font, fill=(0, 0, 0))
                    warning_mess.append(["RightHand_Normal", 'Normal', "Right Hand Position", str(1.0), 'Normal'])
                else:
                    draw_res_err_box.text((rightarm_err_mess_coor[0], rightarm_err_mess_coor[1]), '右手持弓: 正常',
                                          font=result_font, fill=(0, 0, 0))
                    warning_mess.append(["RightHand_Normal", 'Normal', "Right Hand Position", str(1.0), 'Normal'])

                # A-L Error Left Hand ================================================================================ 3
                lefthand_err_mess_coor = [rightarm_err_mess_coor[0],
                                          rightarm_err_mess_coor[1] + int(err_mess_frame[0] // div_space_val)]
                if 0 >= P_var:
                    # print('Right Arm')
                    draw_res_err_box.text((lefthand_err_mess_coor[0], lefthand_err_mess_coor[1]),
                                          leftHand_AL1_ClassName, font=result_font, fill=(0, 0, 255))
                    # warning_mess.append(["E31", str(rightArm_E31), 'Right Hand Hand Position', str(rightArm_E31),
                    #                      'E31-Wrong RH thumb position'])
                    # draw_res_bow.rectangle(ori_right_hand_rectangle_shape, outline="blue", fill=None, width=4)
                elif 0 >= P_var:
                    # print('Else Right Arm')
                    draw_res_err_box.text((lefthand_err_mess_coor[0], lefthand_err_mess_coor[1]),
                                          leftHand_AL2_ClassName, font=result_font, fill=(0, 0, 255))
                    # warning_mess.append(["E32", str(rightArm_E32), 'Right Hand Hand Position', str(rightArm_E32),
                    #                      'E32-Wrong RH index finger position'])
                    # draw_res_bow.rectangle(ori_right_hand_rectangle_shape, outline="blue", fill=None, width=4)
                # elif lefthand_NHand >= P_var:
                #     # print('Else Right Arm')
                #     draw_res_err_box.text((lefthand_err_mess_coor[0], lefthand_err_mess_coor[1]), '左手按弦: 正常',
                #                           font=result_font, fill=(0, 0, 0))
                #     warning_mess.append(["LeftHand_Normal", 'Normal', "Left Hand Position", str(1.0), 'Normal'])
                else:
                    # print('Else Right Arm')
                    draw_res_err_box.text((lefthand_err_mess_coor[0], lefthand_err_mess_coor[1]), '左手按弦: 正常',
                                          font=result_font, fill=(0, 0, 0))
                    # warning_mess.append(["LeftHand_Normal", 'Normal', "Left Hand Position", str(1.0), 'Normal'])

                # E34/E35 Error Right Arm ================================================================================ 3
                if rightArm_E34 >= P_var:
                    # print('Else Right Arm')
                    # draw_res_bow.text((10, 200), rightArm_E34_ClassName + ':' + str(rightArm_E34), font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["E34", str(rightArm_E34), 'Right Arm Position', str(rightArm_E34),
                                         'E34-Right wrist position too inward'])
                    # draw_res_bow.rectangle(ori_right_arm_rectangle_shape, outline="blue", fill=None, width=4)
                elif rightArm_E35 >= P_var:
                    # print('Else Right Arm')
                    # draw_res_bow.text((10, 200), rightArm_E35_ClassName + ':' + str(rightArm_E35), font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["E35", str(rightArm_E35), 'Right Arm Position', str(rightArm_E35),
                                         'E35-Right wrist position too outward'])
                    # draw_res_bow.rectangle(ori_right_arm_rectangle_shape, outline="blue", fill=None, width=4)
                elif rightArm_NHand >= P_var:
                    # print('Else Right Arm')
                    # draw_res_bow.text((10, 200), rightArm_Normal_ClassName, font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["RightArm_Normal", 'Normal', "Right Arm Position", str(1.0), 'Normal'])
                else:
                    # print('Else Right Arm')
                    # draw_res_bow.text((10, 200), rightArm_Normal_ClassName, font=result_font, fill=(b, g, r, a))
                    warning_mess.append(["RightArm_Normal", 'Normal', "Right Arm Position", str(1.0), 'Normal'])

                # # E21/E22/E23 Error Left Arm ================================================================================= 4
                # # draw_res_bow.text((10, 230), "===== Left Arm =====", font=result_font, fill=(b, g, r, a))
                # ori_left_arm_rectangle_shape = [ori_left_arm_point[0], ori_left_arm_point[1]]
                # ori_left_hand_rectangle_shape = [ori_left_hand_point[0], ori_left_hand_point[1]]
                # # if leftArm_E21 >= P_var:
                # if is_left_arm_too_high == True :
                #     # print('Left Arm')
                #     # draw_res_bow.text((10, 260), leftArm_E21_ClassName + ':' + str(leftArm_E21), font=result_font, fill=(b, g, r, a))
                #     warning_mess.append(["E21", str(degree_lh_shoulder_elbow), 'Left Hand Arm Position', str(degree_lh_shoulder_elbow),
                #                          'E21-Left elbow too Hight'])
                #     # draw_res_bow.rectangle(ori_left_arm_rectangle_shape, outline='blue', fill=None, width=4)
                #     try:
                #         r_circle = 4
                #         draw_res_bow.ellipse(
                #             [(lh_shoulder_coor[0] - r_circle, lh_shoulder_coor[1] - r_circle),
                #              (lh_shoulder_coor[0] + r_circle, lh_shoulder_coor[1] + r_circle)],
                #             fill='blue')
                #         draw_res_bow.line([(lh_shoulder_coor[0], lh_shoulder_coor[1]),
                #                            (lh_finger_wrist_elbow_coor[2][0], lh_finger_wrist_elbow_coor[2][1])],
                #                           fill='blue', width=4)
                #     except:
                #         print('Missing Mediapipe')
                # # elif leftArm_E22 >= P_var:
                # elif is_left_arm_too_low == True:
                #     # print('Else Left Arm')
                #     # draw_res_bow.text((10, 260), leftArm_E22_ClassName + ':' + str(leftArm_E22), font=result_font, fill=(b, g, r, a))
                #     warning_mess.append(["E22", str(degree_lh_shoulder_elbow), 'Left Hand Arm Position', str(degree_lh_shoulder_elbow), 'E22-Left elbow too Low'])
                #     # draw_res_bow.rectangle(ori_left_arm_rectangle_shape, outline='blue', fill=None, width=4)
                #     try:
                #         r_circle = 4
                #         draw_res_bow.ellipse(
                #             [(lh_shoulder_coor[0] - r_circle, lh_shoulder_coor[1] - r_circle),
                #              (lh_shoulder_coor[0] + r_circle, lh_shoulder_coor[1] + r_circle)],
                #             fill='blue')
                #         # print(lh_shoulder_coor[0])
                #         # print(lh_shoulder_coor[1])
                #         draw_res_bow.line([(lh_shoulder_coor[0], lh_shoulder_coor[1]),
                #                            (lh_finger_wrist_elbow_coor[2][0], lh_finger_wrist_elbow_coor[2][1])],
                #                           fill='blue', width=4)
                #     except:
                #         print('Missing Mediapipe')
                # elif is_left_hand_slope == True:
                #     # print('Else Left Arm')
                #     # draw_res_bow.text((10, 260), leftArm_E23_ClassName + ':' + str(leftArm_E23), font=result_font, fill=(b, g, r, a))
                #     warning_mess.append(["E23", str(lh_slope_value), 'Left Hand Wrist Position', str(lh_slope_value),
                #                          'E23-Left elbow and wrist in a line'])
                #     # draw_res_bow.rectangle(ori_left_hand_rectangle_shape, outline='blue', fill=None, width=4)
                #     # print((lh_finger_wrist_elbow_coor[0][0], lh_finger_wrist_elbow_coor[0][1]), (lh_finger_wrist_elbow_coor[1][0], lh_finger_wrist_elbow_coor[1][1]), (lh_finger_wrist_elbow_coor[2][0], lh_finger_wrist_elbow_coor[2][1]))
                #     try:
                #         r_circle = 4
                #         draw_res_bow.ellipse(
                #             [(lh_finger_wrist_elbow_coor[0][0] - r_circle, lh_finger_wrist_elbow_coor[0][1] - r_circle),
                #              (
                #              lh_finger_wrist_elbow_coor[0][0] + r_circle, lh_finger_wrist_elbow_coor[0][1] + r_circle)],
                #             fill='blue')
                #         draw_res_bow.ellipse(
                #             [(lh_finger_wrist_elbow_coor[1][0] - r_circle, lh_finger_wrist_elbow_coor[1][1] - r_circle),
                #              (
                #              lh_finger_wrist_elbow_coor[1][0] + r_circle, lh_finger_wrist_elbow_coor[1][1] + r_circle)],
                #             fill='blue')
                #         draw_res_bow.ellipse(
                #             [(lh_finger_wrist_elbow_coor[2][0] - r_circle, lh_finger_wrist_elbow_coor[2][1] - r_circle),
                #              (
                #              lh_finger_wrist_elbow_coor[2][0] + r_circle, lh_finger_wrist_elbow_coor[2][1] + r_circle)],
                #             fill='blue')
                #         draw_res_bow.line([(lh_finger_wrist_elbow_coor[0][0], lh_finger_wrist_elbow_coor[0][1]),
                #                            (lh_finger_wrist_elbow_coor[1][0], lh_finger_wrist_elbow_coor[1][1])],
                #                           fill='blue', width=4)
                #         draw_res_bow.line([(lh_finger_wrist_elbow_coor[1][0], lh_finger_wrist_elbow_coor[1][1]),
                #                            (lh_finger_wrist_elbow_coor[2][0], lh_finger_wrist_elbow_coor[2][1])],
                #                           fill='blue', width=4)
                #     except:
                #         print('Body Landmark Missing')
                # elif leftArm_N >= P_var:
                #     # print('Else Left Arm')
                #     # draw_res_bow.text((10, 260), leftArm_Normal_ClassName, font=result_font, fill=(b, g, r, a))
                #     # warning_mess.append(["LeftArm_Normal", "Normal", 'Left Hand Arm Position', str(1.0), 'Normal'])
                #     warning_mess.append(["LeftHand_Normal", "Normal", 'Left Hand Arm is Inline', str(1.0), 'Normal'])
                # else:
                #     # print('Else Left Arm')
                #     # draw_res_bow.text((10, 260), leftArm_Normal_ClassName, font=result_font, fill=(b, g, r, a))
                #     # warning_mess.append(["LeftArm_Normal", "Normal", 'Left Hand Arm Position', str(1.0), 'Normal'])
                #     warning_mess.append(["LeftHand_Normal", "Normal", 'Left Hand Arm is Inline', str(1.0), 'Normal'])


                # # if leftArm_E23 >= P_var:
                # if is_left_hand_slope == True:
                #     # print('Else Left Arm')
                #     # draw_res_bow.text((10, 260), leftArm_E23_ClassName + ':' + str(leftArm_E23), font=result_font, fill=(b, g, r, a))
                #     warning_mess.append(["E23", str(lh_slope_value), 'Left Hand Wrist Position', str(lh_slope_value),
                #                          'E23-Left elbow and wrist in a line'])
                #     # draw_res_bow.rectangle(ori_left_hand_rectangle_shape, outline='blue', fill=None, width=4)
                #     # print((lh_finger_wrist_elbow_coor[0][0], lh_finger_wrist_elbow_coor[0][1]), (lh_finger_wrist_elbow_coor[1][0], lh_finger_wrist_elbow_coor[1][1]), (lh_finger_wrist_elbow_coor[2][0], lh_finger_wrist_elbow_coor[2][1]))
                #     try:
                #         r_circle = 4
                #         draw_res_bow.ellipse(
                #             [(lh_finger_wrist_elbow_coor[0][0] - r_circle, lh_finger_wrist_elbow_coor[0][1] - r_circle),
                #              (lh_finger_wrist_elbow_coor[0][0] + r_circle, lh_finger_wrist_elbow_coor[0][1] + r_circle)],
                #             fill='blue')
                #         draw_res_bow.ellipse(
                #             [(lh_finger_wrist_elbow_coor[1][0] - r_circle, lh_finger_wrist_elbow_coor[1][1] - r_circle),
                #              (lh_finger_wrist_elbow_coor[1][0] + r_circle, lh_finger_wrist_elbow_coor[1][1] + r_circle)],
                #             fill='blue')
                #         draw_res_bow.ellipse(
                #             [(lh_finger_wrist_elbow_coor[2][0] - r_circle, lh_finger_wrist_elbow_coor[2][1] - r_circle),
                #              (lh_finger_wrist_elbow_coor[2][0] + r_circle, lh_finger_wrist_elbow_coor[2][1] + r_circle)],
                #             fill='blue')
                #         draw_res_bow.line([(lh_finger_wrist_elbow_coor[0][0], lh_finger_wrist_elbow_coor[0][1]),
                #                            (lh_finger_wrist_elbow_coor[1][0], lh_finger_wrist_elbow_coor[1][1])],
                #                           fill='blue', width=4)
                #         draw_res_bow.line([(lh_finger_wrist_elbow_coor[1][0], lh_finger_wrist_elbow_coor[1][1]),
                #                            (lh_finger_wrist_elbow_coor[2][0], lh_finger_wrist_elbow_coor[2][1])],
                #                           fill='blue', width=4)
                #     except:
                #         print('Body Landmark Missing')
                # else:
                #     # print('Else Left Arm')
                #     # draw_res_bow.text((10, 260), leftArm_Normal_ClassName, font=result_font, fill=(b, g, r, a))
                #     warning_mess.append(["LeftHand_Normal", "Normal", 'Left Hand Arm is Inline', str(1.0), 'Normal'])

                # E15 Error Knees Position ============================================================================= 7
                knee_err_mess_coor = [lefthand_err_mess_coor[0],
                                      lefthand_err_mess_coor[1] + int(err_mess_frame[0] // div_space_val)]
                if is_knees_shoulder == True:
                    draw_res_err_box.text((knee_err_mess_coor[0], knee_err_mess_coor[1]), E15_classname,
                                          font=result_font, fill=(0, 128, 255))
                    warning_mess.append(["E15", str(knees_shoulder_distance), 'Knees Position', str(degree_body),
                                         'E15-Put knees in normal position'])
                    # draw_res_bow.rectangle((hip_left_point, hip_right_point), outline="blue", fill=None, width=4)
                    r_circle = 7
                    draw_res_bow.ellipse(
                        [(lh_knees_coor[0] - r_circle, lh_knees_coor[1] - r_circle),
                         (lh_knees_coor[0] + r_circle, lh_knees_coor[1] + r_circle)],
                        fill=(0, 128, 255))
                    draw_res_bow.ellipse(
                        [(rh_knees_coor[0] - r_circle, rh_knees_coor[1] - r_circle),
                         (rh_knees_coor[0] + r_circle, rh_knees_coor[1] + r_circle)],
                        fill=(0, 128, 255))
                    draw_res_err_box.ellipse(
                        [(lh_knees_coor[0] - r_circle, lh_knees_coor[1] - r_circle),
                         (lh_knees_coor[0] + r_circle, lh_knees_coor[1] + r_circle)],
                        fill=(0, 128, 255))
                    draw_res_err_box.ellipse(
                        [(rh_knees_coor[0] - r_circle, rh_knees_coor[1] - r_circle),
                         (rh_knees_coor[0] + r_circle, rh_knees_coor[1] + r_circle)],
                        fill=(0, 128, 255))
                else:
                    draw_res_err_box.text((knee_err_mess_coor[0], knee_err_mess_coor[1]), '兩膝: 正常', font=result_font,
                                          fill=(0, 0, 0))
                    warning_mess.append(["Knees_Normal", "Normal", 'Knees Position', str(0), 'Normal'])
                resize_img = (256, 256)
                img_resized = np.array(img_pil.resize(resize_img))
                img_original = np.array(img_pil)
                img_err_mess = np.array(img_err)
                # output_array.append([warning_mess, img_original])
                output_array.append([warning_mess])
                output_json_number += 1
                videoOut_1.write(img_original)
                videoOut_2.write(img_err_mess)
                # print('Write Frame -', idx_write_frame)
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
            x_test_left_hand_point = []
            x_test_all_body_point = []
            x_test_erhu_line_point = []
            x_test_bow_line_point = []
            x_test_bow_point = []
            # break
        # else:
        #     frame_number += 1
    # print(output_array)
    # np.savez_compressed(os.path.join(result_folder, filename + "_" + curr_time), output_array)

    if len(x_test_video_original) > 0:
        indx = 0
        for img in x_test_video_original:
            indx += 1
            print('Last Frame:', str(indx))
            warning_mess = []
            warning_mess.append(["Head_Normal", "Normal", 'Head Position', str(0), 'Normal'])
            warning_mess.append(["Body_Normal", "Normal", 'Body Position', str(0), 'Normal'])
            warning_mess.append(['Shoulder_Normal', 'Normal', 'Shoulders Position', str(0), 'Normal'])
            warning_mess.append(["RightHand_Normal", 'Normal', "Right Hand Position", str(1.0), 'Normal'])
            warning_mess.append(["RightArm_Normal", 'Normal', "Right Arm Position", str(1.0), 'Normal'])
            warning_mess.append(["LeftHand_Normal", "Normal", 'Left Hand Arm is Inline', str(1.0), 'Normal'])
            warning_mess.append(["Erhu_Normal", "Normal", "Erhu Position", str(1.0), "Normal"])
            warning_mess.append(["Bow_Normal", "Normal", "Bow Erhu Position", str(1.0), "Normal"])
            warning_mess.append(["Knees_Normal", "Normal", 'Knees Position', str(0), 'Normal'])
            output_array.append([warning_mess])
            output_json_number += 1
            videoOut_1.write(img)

    np.savez_compressed(os.path.join(result_folder, filename), output_array)

    # Save JSON ==========
    np_array_to_list = output_array
    json_file = os.path.join(result_folder, filename + '.json')
    json.dump(np_array_to_list, codecs.open(json_file, 'w', encoding='utf-8'), sort_keys=True, indent=4)
    # ====================

    videoInput.release()
    tensorflow.keras.backend.clear_session()
    videoOut_1.release()
    videoOut_2.release()
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
    print("Json Count=", output_json_number)
    print("Frame Count=", frame_number)
    print("Frame Count Failed=", frame_failed)
    return os.path.join(result_folder, filename + ".mp4"), os.path.join(result_folder, filename + "_err_msg.mp4")

# main_predict('/home/minelab/dev/erhu-project/Data Test/E33_5072_clip_0.mp4')
# main_predict('/home/minelab/dev/erhu-project/Data Test/E32_clip_0.mp4')