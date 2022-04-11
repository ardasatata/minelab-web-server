import argparse
import os
import cv2
import imutils
import numpy as np
import mediapipe as mp
from math import degrees, atan2
from pixellib.instance import custom_segmentation
from datetime import datetime

def get_contours(frame):
    image   = frame.copy()
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray    = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh  = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh  = cv2.erode(thresh, None, iterations=2)
    thresh  = cv2.dilate(thresh, None, iterations=2)
    cnts    = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts    = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    return c

def get_angle(p1, p2):
    xDiff   = p2[0] - p1[0]
    yDiff   = p2[1] - p1[1]
    angles  = degrees(atan2(yDiff, xDiff))
    return round(angles, 2)

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
    extLeft         = tuple(c[c[:, :, 0].argmin()][0])
    extRight        = tuple(c[c[:, :, 0].argmax()][0])
    extTop          = tuple(c[c[:, :, 1].argmin()][0])
    extBot          = tuple(c[c[:, :, 1].argmax()][0])
    colorGreen          = (0, 255, 0)
    colorRed            = (0, 0, 255)
    colorBlue           = (255, 0, 0)
    colorYellow         = (0, 255, 255)
    colorPurple         = (128, 0, 128)
    thickness           = 2
    centerPointTop      = (extLeft[0] + (extRight[0] - extLeft[0]) // 2, extTop[1])
    centerPointBottom   = (extLeft[0] + (extRight[0] - extLeft[0]) // 2, extBot[1])
    centerLengthBottom  = centerPointBottom[1] - centerPointTop[1]
    topQinZhenLength    = int((centerLengthBottom * 12.2) / 100)
    mark_sectionLength  = int((centerLengthBottom * 0.75) / 100)
    bottomQinZhenLength = int((centerLengthBottom * 20.0) / 100)
    middleErhuLength    = int((centerLengthBottom * 52.87) / 100)
    # Draw to get the top Qin Zhen
    topQinZhenStart         = centerPointTop
    topQinZhenEnd           = (centerPointTop[0], centerPointTop[1] + topQinZhenLength)
    rightTopQinZhenDotStart = (extRight[0], extTop[1])
    leftTopQinZhenDotStart  = (extLeft[0], extTop[1])
    leftTopQinZhenDotEnd    = (extLeft[0], extTop[1] + topQinZhenLength)
    rightTopQinZhenDotEnd   = (extRight[0], extTop[1] + topQinZhenLength)
    # cv2.rectangle(img_ori, leftTopQinZhenDotStart, rightTopQinZhenDotEnd, colorPurple, thickness)
    # === Remark for draw all segment of erhu ======================================================================
    mask1 = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask1, leftTopQinZhenDotStart, rightTopQinZhenDotEnd, 255, -1)
    mask_hand1 = cv2.bitwise_and(image, image, mask=mask1)
    # ==============================================================================================================
    # Draw to get the marker line point ============================================================================
    MarkerStart = topQinZhenEnd
    MarkerEnd   = (MarkerStart[0], MarkerStart[1] + mark_sectionLength)
    rightMarkerDotStart = rightTopQinZhenDotEnd
    leftMarkerDotStart  = leftTopQinZhenDotEnd
    leftMarkerDotEnd    = (leftMarkerDotStart[0], leftMarkerDotStart[1] + mark_sectionLength)
    rightMarkerDotEnd   = (rightMarkerDotStart[0], rightMarkerDotStart[1] + mark_sectionLength)
    # cv2.rectangle(img_ori, leftTopQinZhenDotStart, rightTopQinZhenDotEnd, colorPurple, thickness)
    # === Remark for draw all segment of erhu ======================================================================
    mask_marker = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask_marker, leftMarkerDotStart, rightMarkerDotEnd, 255, -1)
    mask_markers = cv2.bitwise_and(image, image, mask=mask_marker)
    # ==============================================================================================================
    # === Draw local boundary for Marker Line Point ================================================================
    MarkerLine  = mask_markers
    gray        = cv2.cvtColor(MarkerLine, cv2.COLOR_BGR2GRAY)
    gray        = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh      = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh      = cv2.erode(thresh, None, iterations=2)
    thresh      = cv2.dilate(thresh, None, iterations=2)
    cnts        = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts        = imutils.grab_contours(cnts)
    c           = max(cnts, key=cv2.contourArea)
    extLeftMarker   = tuple(c[c[:, :, 0].argmin()][0])
    extRightMarker  = tuple(c[c[:, :, 0].argmax()][0])
    extTopMarker    = tuple(c[c[:, :, 1].argmin()][0])
    extBotMarker    = tuple(c[c[:, :, 1].argmax()][0])
    rightTopDot_Marker      = (extRightMarker[0], extTopMarker[1])
    leftTopDot_Marker       = (extLeftMarker[0], extTopMarker[1])
    leftBottomDot_Marker    = (extLeftMarker[0], extBotMarker[1])
    rightBottomDot_Marker   = (extRightMarker[0], extBotMarker[1])
    topMiddlePoint_Marker   = (leftTopDot_Marker[0] + ((rightTopDot_Marker[0] - leftTopDot_Marker[0]) // 2), rightTopDot_Marker[1])
    # ==================================================================================================================
    # Draw to get the bottom QinZhen
    bottomQinZhenStart      = MarkerEnd
    bottomQinZhenEnd        = (bottomQinZhenStart[0], bottomQinZhenStart[1] + bottomQinZhenLength)
    rightBotQinZhenDotStart = rightMarkerDotEnd
    leftBotQinZhenDotStart  = leftMarkerDotEnd
    leftBotQinZhenDotEnd    = (leftBotQinZhenDotStart[0], leftBotQinZhenDotStart[1] + bottomQinZhenLength)
    rightBotQinZhenDotEnd   = (rightBotQinZhenDotStart[0], rightBotQinZhenDotStart[1] + bottomQinZhenLength)
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
    extLeftBotQinZhen   = tuple(c[c[:, :, 0].argmin()][0])
    extRightBotQinZhen  = tuple(c[c[:, :, 0].argmax()][0])
    extTopBotQinZhen    = tuple(c[c[:, :, 1].argmin()][0])
    extBotBotQinZhen    = tuple(c[c[:, :, 1].argmax()][0])
    rightTopDot_BotQinZhen      = (extRightBotQinZhen[0], extTopBotQinZhen[1])
    leftTopDot_BotQinZhen       = (extLeftBotQinZhen[0], extTopBotQinZhen[1])
    leftBottomDot_BotQinZhen    = (extLeftBotQinZhen[0], extBotBotQinZhen[1])
    rightBottomDot_BotQinZhen   = (extRightBotQinZhen[0], extBotBotQinZhen[1])
    topMiddlePoint_BotQinZhen   = (leftTopDot_BotQinZhen[0] + ((rightTopDot_BotQinZhen[0] - leftTopDot_BotQinZhen[0]) // 2), rightTopDot_BotQinZhen[1])
    cv2.rectangle(img_ori, leftTopDot_BotQinZhen, rightBottomDot_BotQinZhen, colorGreen, thickness)
    QinZhenCropped = image[leftTopDot_BotQinZhen[1]:rightBottomDot_BotQinZhen[1], leftTopDot_BotQinZhen[0]:rightBottomDot_BotQinZhen[0]]
    # Draw to get the middle of Erhu ===============================================================================
    middleErhuStart         = bottomQinZhenEnd
    middleErhuEnd           = (middleErhuStart[0], middleErhuStart[1] + middleErhuLength)
    rightMidErhuDotStart    = rightBotQinZhenDotEnd
    leftMidErhuDotStart     = leftBotQinZhenDotEnd
    leftMidErhuDotEnd       = (leftMidErhuDotStart[0], leftMidErhuDotStart[1] + middleErhuLength)
    rightMidErhuDotEnd      = (rightMidErhuDotStart[0], rightMidErhuDotStart[1] + middleErhuLength)
    # cv2.rectangle(img_ori, leftMidErhuDotStart, rightMidErhuDotEnd, colorPurple, thickness)
    # === Remark for draw all segment of erhu ======================================================================
    mask3 = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask3, leftMidErhuDotStart, rightMidErhuDotEnd, 255, -1)
    mask_hand3 = cv2.bitwise_and(image, image, mask=mask3)
    # ==============================================================================================================
    # Draw to get the Qin Thong
    QinThongStart           = middleErhuEnd
    QinThongEnd             = centerPointBottom
    rightQinThongDotStart   = rightMidErhuDotEnd
    leftQinThongDotStart    = leftMidErhuDotEnd
    leftQinThongDotEnd      = (leftQinThongDotStart[0], centerPointBottom[1])
    rightQinThongDotEnd     = (rightQinThongDotStart[0], centerPointBottom[1])
    # cv2.rectangle(img_ori, leftQinThongDotStart, rightQinThongDotEnd, colorGreen, thickness)
    # === Remark for draw all segment of erhu ============================================
    mask4 = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask4, leftQinThongDotStart, rightQinThongDotEnd, 255, -1)
    mask_hand4 = cv2.bitwise_and(image, image, mask=mask4)
    # ====================================================================================
    # === Draw local boundary for Qin Thong ======================================================================
    QinThong    = mask_hand4
    gray        = cv2.cvtColor(QinThong[:, :], cv2.COLOR_BGR2GRAY)
    gray        = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh      = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh      = cv2.erode(thresh, None, iterations=2)
    thresh      = cv2.dilate(thresh, None, iterations=2)
    cnts        = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts        = imutils.grab_contours(cnts)
    c           = max(cnts, key=cv2.contourArea)
    extLeftQinThong     = tuple(c[c[:, :, 0].argmin()][0])
    extRightQinThong    = tuple(c[c[:, :, 0].argmax()][0])
    extTopQinThong      = tuple(c[c[:, :, 1].argmin()][0])
    extBotQinThong      = tuple(c[c[:, :, 1].argmax()][0])
    rightTopDot_QinThong        = (extRightQinThong[0], extTopQinThong[1])
    leftTopDot_QinThong         = (extLeftQinThong[0], extTopQinThong[1])
    leftBottomDot_QinThong      = (extLeftQinThong[0], extBotQinThong[1])
    rightBottomDot_QinThong     = (extRightQinThong[0], extBotQinThong[1])
    bottomMiddlePoint_QinThong  = (leftBottomDot_QinThong[0] + ((rightBottomDot_QinThong[0] - leftBottomDot_QinThong[0]) // 3), rightBottomDot_QinThong[1])
    cv2.rectangle(img_ori, leftTopDot_QinThong, rightBottomDot_QinThong, colorGreen, thickness)
    QinThongCropped = image[leftTopDot_QinThong[1]:rightBottomDot_QinThong[1], leftTopDot_QinThong[0]:rightBottomDot_QinThong[0]]
    cv2.line(img_ori, topMiddlePoint_Marker, bottomMiddlePoint_QinThong, colorRed, 5)
    erhu_line = [[topMiddlePoint_Marker[0], topMiddlePoint_Marker[1]], [bottomMiddlePoint_QinThong[0], bottomMiddlePoint_QinThong[1]]]
    # ==============================================================================================================
    # cv2.imshow("Image", img_ori)
    # cv2.waitKey(1)
    return img_ori, erhu_line, QinZhenCropped, QinThongCropped

def calculate_area(p1, p2):
    p1x = p1[0]
    p1y = p1[1]
    p2x = p2[0]
    p2y = p2[1]
    if p1x > p2x :
        x_length = p1x - p2x
    else:
        x_length = p2x - p1x
    if p1y > p2y :
        y_length = p1y - p2y
    else:
        y_length = p2y - p1y
    value_of_area = x_length * y_length

    return value_of_area

def get_body_ratio(img):
    image   = img.copy()
    gray    = cv2.cvtColor(image[:, :], cv2.COLOR_BGR2GRAY)
    gray    = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh  = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh  = cv2.erode(thresh, None, iterations=2)
    thresh  = cv2.dilate(thresh, None, iterations=2)
    cnts    = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts    = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeftImage    = tuple(c[c[:, :, 0].argmin()][0])
    extRightImage   = tuple(c[c[:, :, 0].argmax()][0])
    extTopImage     = tuple(c[c[:, :, 1].argmin()][0])
    extBotImage     = tuple(c[c[:, :, 1].argmax()][0])
    rightTopImage   = (extRightImage[0], extTopImage[1])
    leftTopImage    = (extLeftImage[0], extTopImage[1])
    leftBottomImage = (extLeftImage[0], extBotImage[1])
    rightBottomImage = (extRightImage[0], extBotImage[1])
    widthImage  = image.shape[1]
    heightImage = image.shape[0]
    # print(widthImage, heightImage)
    value_body_areas    = calculate_area(leftTopImage, rightBottomImage)
    value_img_areas     = heightImage * widthImage
    # print(value_img_areas)
    percent_areas       = round((value_body_areas/value_img_areas) * 100)
    # print(percent_areas)
    cv2.rectangle(image, leftTopImage, rightBottomImage, (0, 255, 0), 2)
    cv2.putText(image, 'L='+str(value_body_areas) + ', Percent = ' + str(percent_areas),leftTopImage, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    return image

def get_var_multiplier(w,h):
    # print('res:',w,h)
    if w <= 640 and h <= 480 :
        var_multiplier = 0.5
        # print('640x480')
    elif (w > 640 and w <= 1280) and (h > 480 and h <= 720) :
        var_multiplier = 1
        # print('1280x720')
    elif (w > 1280 and w <= 1920) and (h > 720 and h <= 1080) :
        var_multiplier = 1.8
        # print('1920x1080')
    else :
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

def body_segment(img, frame):
    mp_drawing          = mp.solutions.drawing_utils
    mp_drawing_styles   = mp.solutions.drawing_styles
    mp_holistic         = mp.solutions.holistic

    image       = frame.copy()
    img_ori     = img.copy()
    image_ori   = img.copy()
    colorGreen  = (0, 255, 0)
    colorRed    = (0, 0, 255)
    colorBlue   = (255, 0, 0)
    colorYellow = (0, 255, 255)
    thickness   = 2
    length_of_hand_boundaries = 50
    length_of_arm_boundaries_1 = [120, 80]
    length_of_arm_boundaries_2 = [30, 180]
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
    rightTopDot_Body    = (extRightBody[0], extTopBody[1])
    leftTopDot_Body     = (extLeftBody[0], extTopBody[1])
    leftBottomDot_Body  = (extLeftBody[0], extBotBody[1])
    rightBottomDot_Body = (extRightBody[0], extBotBody[1])
    if rightTopDot_Body[0] > leftTopDot_Body[0]:
        middleTopBody       = (leftTopDot_Body[0] + ((rightTopDot_Body[0] - leftTopDot_Body[0])//2), rightTopDot_Body[1])
        # middleBottomBody    = (leftBottomDot_Body[0] + ((rightBottomDot_Body[0] - leftBottomDot_Body[0])//2), rightBottomDot_Body[1])
        halfRectangleBodyCoordinate = (middleTopBody, rightBottomDot_Body)
    else:
        middleTopBody = (rightTopDot_Body[0] + ((leftTopDot_Body[0] - rightTopDot_Body[0]) // 2), leftTopDot_Body[1])
        # middleBottomBody = (rightBottomDot_Body[0] + ((leftBottomDot_Body[0] - rightBottomDot_Body[0]) // 2), leftBottomDot_Body[1])
        halfRectangleBodyCoordinate = (middleTopBody, leftBottomDot_Body)

    with mp_holistic.Holistic( min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
        # print('Holistic Segment')
        image.flags.writeable = False
        image   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image_height, image_width, _ = image.shape
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        x_R1 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].x * image_width + round(length_of_hand_boundaries * var_multiplier))
        x_R2 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].y * image_height - round(length_of_hand_boundaries * var_multiplier))
        x_R3 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].x * image_width - round(length_of_hand_boundaries * var_multiplier))
        x_R4 = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_THUMB].y * image_height + round(length_of_hand_boundaries * var_multiplier))
        if x_R1 > image_width: x_R1 = image_width
        if x_R2 < 0: x_R2 = 0
        if x_R3 < 0: x_R3 = 0
        if x_R4 > image_height: x_R4 = image_height
        # Left hand wrist/shoulder rectangle point
        x_L1  = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width) + round(length_of_arm_boundaries_1[0] * var_multiplier)
        x_L2  = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height) - round(length_of_arm_boundaries_1[1] * var_multiplier)
        x_L3  = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width) - round(length_of_arm_boundaries_2[0] * var_multiplier)
        x_L4  = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height) + round(length_of_arm_boundaries_2[1] * var_multiplier)
        cv2.circle(img_ori,(x_L1, x_L2), 5, colorRed, 1)
        cv2.circle(img_ori, (x_L3, x_L4), 5, colorBlue, 1)
        # Left hand thumb rectangle point
        x_L1a = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_THUMB].x * image_width) + round(length_of_hand_boundaries * var_multiplier)
        x_L2a = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_THUMB].y * image_height) - round(length_of_hand_boundaries * var_multiplier)
        x_L3a = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_THUMB].x * image_width) - round(length_of_hand_boundaries * var_multiplier)
        x_L4a = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_THUMB].y * image_height) + round(length_of_hand_boundaries * var_multiplier)
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

        x_L1_shoulders = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width)
        x_L2_shoulders = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height)
        x_R1_shoulders = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width)
        x_R2_shoulders = round(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_height)

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

        if x_R1_shoulders > x_L1_shoulders :
            middle_shoulder_point = (((x_R1_shoulders - x_L1_shoulders)//2) + x_L1_shoulders, x_L2_shoulders)
        else:
            middle_shoulder_point = (((x_L1_shoulders - x_R1_shoulders)//2) + x_R1_shoulders, x_L2_shoulders)
        if x_R1_hip > x_L1_hip :
            middle_hip_point = (((x_R1_hip - x_L1_hip)//2) + x_L1_hip, x_L2_hip)
        else:
            middle_hip_point = (((x_L1_hip - x_R1_hip)//2) + x_R1_hip, x_L2_hip)

        if x_R1_eye > x_L1_eye:
            middle_eye_point = (((x_R1_eye - x_L1_eye) // 2) + x_L1_eye, x_L2_eye)
        else:
            middle_eye_point = (((x_L1_eye - x_R1_eye) // 2) + x_R1_eye, x_L2_eye)
        if x_R1_mouth > x_L1_mouth:
            middle_mouth_point = (((x_R1_mouth - x_L1_mouth) // 2) + x_L1_mouth, x_L2_mouth)
        else:
            middle_mouth_point = (((x_L1_mouth - x_R1_mouth) // 2) + x_R1_mouth, x_L2_mouth)

        right_arm           = cv2.rectangle(img_ori, (x_L1, x_L2), (x_L3, x_L4),  colorBlue, thickness)
        right_arm_cropped   = cropped_by_pixel(image, x_L1, x_L2, x_L3, x_L4)
        # cv2.imshow('left_hand_cropped', left_hand_cropped)

        right_hand       = cv2.rectangle(right_arm, (x_R1, x_R2), (x_R3, x_R4), colorGreen, thickness)
        right_hand_cropped = cropped_by_pixel(image, x_R1, x_R2, x_R3, x_R4)
        # cv2.imshow('right_arm_cropped', right_arm_cropped)

        left_hand          = cv2.rectangle(right_hand, (x_L1a, x_L2a), (x_L3a, x_L4a), colorRed, thickness)
        left_hand_cropped  = cropped_by_pixel(image, x_L1a, x_L2a, x_L3a, x_L4a)
        # cv2.imshow('right_hand_cropped', right_hand_cropped)

        left_hip        = cv2.circle(left_hand, (x_L1_hip, x_L2_hip), 10, colorRed, -1)
        right_hip       = cv2.circle(left_hip, (x_R1_hip, x_R2_hip), 10, colorRed, -1)
        left_shoulders  = cv2.circle(right_hip, (x_L1_shoulders, x_L2_shoulders), 10, colorRed, -1)
        right_shoulders = cv2.circle(left_shoulders, (x_R1_shoulders, x_R2_shoulders), 10, colorRed, -1)
        middle_shoulder = cv2.circle(right_shoulders, middle_shoulder_point, 1, colorRed, -1)
        middle_hip      = cv2.circle(middle_shoulder, middle_hip_point, 1, colorRed, -1)
        face_rectangle  = cv2.rectangle(middle_hip, (x_L1f, x_L2f), (x_L3f, x_L4f), colorRed, thickness)
        draw_body_line  = cv2.line(face_rectangle, middle_shoulder_point, middle_hip_point, colorRed, 2)
        draw_face_line  = cv2.line(draw_body_line, middle_eye_point, middle_mouth_point, colorRed, 2)
        draw_shoulder_line      = cv2.line(draw_face_line, (x_L1_shoulders, x_L2_shoulders), (x_R1_shoulders, x_R2_shoulders), colorRed, 2)
        draw_ear_face_line      = cv2.line(draw_shoulder_line, (x_L1_ear,x_L2_ear), (x_R1_ear, x_R2_ear), colorRed, thickness)
        draw_left_knees_circle  = cv2.circle(draw_ear_face_line, (x_L1_knees, x_L2_knees), 10, colorRed, -1)
        draw_right_knees_circle = cv2.circle(draw_left_knees_circle, (x_R1_knees, x_R2_knees), 10, colorRed, -1)
        degrees_elbow           = get_angle(middle_shoulder_point, middle_hip_point)
        degrees_shoulder        = get_angle((x_R1_shoulders, x_R2_shoulders),(x_L1_shoulders, x_L2_shoulders))
        degrees_ear_face        = get_angle((x_R1_ear, x_R2_ear),(x_L1_ear,x_L2_ear))
        if degrees_ear_face == 0 :
            cv2.putText(draw_right_knees_circle, '[E11] Head Angle : Normal', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            cv2.putText(draw_right_knees_circle, '[E11] Head Angle : ' + str(degrees_ear_face), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        if degrees_elbow == 90 :
            cv2.putText(draw_right_knees_circle, '[E14] Body Angle : Normal', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            cv2.putText(draw_right_knees_circle, '[E14] Body Angle : ' + str(degrees_elbow), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(draw_right_knees_circle, '[E12-E13] Shoulder Angle : ' + str(degrees_shoulder), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        if degrees_shoulder > 0 :
            cv2.putText(draw_right_knees_circle, '- [E12] Right Shoulder  : Too Hight', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(draw_right_knees_circle, '- [E13] Left Shoulder : Too Low', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        elif degrees_shoulder < 0 :
            cv2.putText(draw_right_knees_circle, '- [E12] Right Shoulder  : Too Low', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(draw_right_knees_circle, '- [E13] Left Shoulder : Too Hight', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else :
            cv2.putText(draw_right_knees_circle, '- [E12] Left Shoulder  : Normal;', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(draw_right_knees_circle, '- [E13] Right Shoulder : Normal', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        halfBodyRectangle = cv2.rectangle(draw_right_knees_circle, halfRectangleBodyCoordinate[0], halfRectangleBodyCoordinate[1], colorBlue, thickness)
        half_body_cropped = cropped_by_pixel(image, halfRectangleBodyCoordinate[0][0], halfRectangleBodyCoordinate[0][1], halfRectangleBodyCoordinate[1][0], halfRectangleBodyCoordinate[1][1])
        ret_image = halfBodyRectangle
    return ret_image, left_hand_cropped, right_hand_cropped, right_arm_cropped, half_body_cropped

def bow_segment(img_ori, frame):
    colorGreen  = (0, 255, 0)
    colorRed    = (0, 0, 255)
    colorBlue   = (255, 0, 0)
    colorYellow = (0, 255, 255)
    thickness   = 5
    image   = frame.copy()
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray    = cv2.GaussianBlur(gray, (5, 5), 0)
    # threshold the image, then perform a series of erosions dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # determine the most extreme points along the contour
    extLeft     = tuple(c[c[:, :, 0].argmin()][0])
    extRight    = tuple(c[c[:, :, 0].argmax()][0])
    extTop      = tuple(c[c[:, :, 1].argmin()][0])
    extBot      = tuple(c[c[:, :, 1].argmax()][0])
    rightTopDot     = (extRight[0], extTop[1])
    leftTopDot      = (extLeft[0], extTop[1])
    leftBottomDot   = (extLeft[0], extBot[1])
    rightBottomDot  = (extRight[0], extBot[1])
    bow_cropped     = image[leftTopDot[1]:rightBottomDot[1], leftTopDot[0]:rightBottomDot[0]]
    BowLength       = leftTopDot[0] - rightTopDot[0]
    topBowLength    = int((BowLength * 5) / 100)
    middleBowLength = int((BowLength * 90) / 100)
    bottomBowLength = int((BowLength * 5) / 100)
    # === Get start point of Bow Lines =========================================================================
    leftTopDot_TopBow   = leftTopDot
    rightTopDot_TopBow  = (leftTopDot[0] - topBowLength, leftTopDot[1])
    leftBotDot_TopBow   = leftBottomDot
    rightBotDot_TopBow  = (leftBottomDot[0] - topBowLength, leftBottomDot[1])
    mask1 = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask1, leftTopDot_TopBow, rightBotDot_TopBow, 255, -1)
    mask_bow1 = cv2.bitwise_and(image, image, mask=mask1)
    bow_1   = mask_bow1.copy()
    gray    = cv2.cvtColor(bow_1, cv2.COLOR_BGR2GRAY)
    gray    = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh  = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh  = cv2.erode(thresh, None, iterations=2)
    thresh  = cv2.dilate(thresh, None, iterations=2)
    cnts    = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts    = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # cv2.drawContours(image, [c], -1, (0, 255, 255), 1)
    extLeftBow1     = tuple(c[c[:, :, 0].argmin()][0])
    extRightBow1    = tuple(c[c[:, :, 0].argmax()][0])
    extTopBow1      = tuple(c[c[:, :, 1].argmin()][0])
    extBotBow1      = tuple(c[c[:, :, 1].argmax()][0])
    rightTopBow1    = (extRightBow1[0], extTopBow1[1])
    leftTopBow1     = (extLeftBow1[0], extTopBow1[1])
    leftBotBow1     = (extLeftBow1[0], extBotBow1[1])
    rightBotBow1    = (extRightBow1[0], extBotBow1[1])
    middlePoint_Bow1 = (((rightBotBow1[0] - leftTopBow1[0]) // 2) + leftTopBow1[0],
                        (((rightBotBow1[1] - leftTopBow1[1]) // 2) + leftTopBow1[1]))
    # === End of Get start point of Bow Lines ==================================================================
    # === Get start point of Bow Lines =========================================================================
    leftTopDot_MidBow   = rightTopDot_TopBow
    rightTopDot_MidBow  = (leftTopDot_MidBow[0] - middleBowLength, leftTopDot_MidBow[1])
    leftBotDot_MidBow   = rightBotDot_TopBow
    rightBotDot_MidBow  = (leftBotDot_MidBow[0] - middleBowLength, leftBotDot_MidBow[1])
    mask2 = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask2, leftTopDot_MidBow, rightBotDot_MidBow, 255, -1)
    mask_bow2 = cv2.bitwise_and(image, image, mask=mask2)
    bow_2   = mask_bow2.copy()
    gray    = cv2.cvtColor(bow_2, cv2.COLOR_BGR2GRAY)
    gray    = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh  = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh  = cv2.erode(thresh, None, iterations=2)
    thresh  = cv2.dilate(thresh, None, iterations=2)
    cnts    = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts    = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # cv2.drawContours(image, [c], -1, (0, 255, 255), 1)
    extLeftBow2     = tuple(c[c[:, :, 0].argmin()][0])
    extRightBow2    = tuple(c[c[:, :, 0].argmax()][0])
    extTopBow2      = tuple(c[c[:, :, 1].argmin()][0])
    extBotBow2      = tuple(c[c[:, :, 1].argmax()][0])
    rightTopBow2    = (extRightBow2[0], extTopBow2[1])
    lefTopBow2      = (extLeftBow2[0], extTopBow2[1])
    leftBotBow2     = (extLeftBow2[0], extBotBow2[1])
    rightBotBow2    = (extRightBow2[0], extBotBow2[1])
    # === End of Get start point of Bow Lines ==================================================================
    # === Get start point of Bow Lines =========================================================================
    leftTopDot_BotBow   = rightTopDot_MidBow
    rightTopDot_BotBow  = (leftTopDot_BotBow[0] - bottomBowLength, leftTopDot_BotBow[1])
    leftBotDot_BotBow   = rightBotDot_MidBow
    rightBotDot_BotBow  = (leftBotDot_BotBow[0] - bottomBowLength, leftBotDot_BotBow[1])
    mask3 = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask3, leftTopDot_BotBow, rightBotDot_BotBow, 255, -1)
    mask_bow2 = cv2.bitwise_and(image, image, mask=mask3)
    bow_2   = mask_bow2.copy()
    gray    = cv2.cvtColor(bow_2, cv2.COLOR_BGR2GRAY)
    gray    = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh  = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    thresh  = cv2.erode(thresh, None, iterations=2)
    thresh  = cv2.dilate(thresh, None, iterations=2)
    cnts    = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts    = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # cv2.drawContours(image, [c], -1, (0, 255, 255), 1)
    extLeftBow3     = tuple(c[c[:, :, 0].argmin()][0])
    extRightBow3    = tuple(c[c[:, :, 0].argmax()][0])
    extTopBow3      = tuple(c[c[:, :, 1].argmin()][0])
    extBotBow3      = tuple(c[c[:, :, 1].argmax()][0])
    rightTopBow3    = (extRightBow3[0], extTopBow3[1])
    leftTopBow3     = (extLeftBow3[0], extTopBow3[1])
    leftBotBow3     = (extLeftBow3[0], extBotBow3[1])
    rightBotBow3    = (extRightBow3[0], extBotBow3[1])
    middlePoint_Bow3 = (((rightBotBow3[0] - leftTopBow3[0]) // 2) + leftTopBow3[0], (((rightBotBow3[1] - leftTopBow3[1]) // 2) + leftTopBow3[1]))
    # === End of Get start point of Bow Lines ==================================================================
    if middlePoint_Bow1[1] < middlePoint_Bow3[1]:
        cv2.line(img_ori, leftTopBow1, rightBotBow3, colorRed, thickness)
        bow_line = [[leftTopBow1[0], leftTopBow1[1]], [rightBotBow3[0], rightBotBow3[1]]]
    else:
        cv2.line(img_ori, leftBotBow1, rightTopBow3, colorRed, thickness)
        bow_line = [[leftBotBow1[0], leftBotBow1[1]], [rightTopBow3[0], rightTopBow3[1]]]

    return img_ori, bow_line, bow_cropped


def _main_(video_path_from_react):
    bow_line = [[0, 0], [0, 0]]
    erhu_line = [[0, 0], [0, 0]]
    print(video_path_from_react)
    start_now = datetime.now()
    dt_string = start_now.strftime("%d-%m-%Y_%H%M%S")
    result_folder = "result/FromReact"
    by_date = os.path.join(result_folder, dt_string)
    if os.path.exists(by_date) == False:
        os.mkdir(by_date)
    result_folder = os.path.join(result_folder, dt_string)
    path_bow = os.path.join(result_folder, 'bowPart')
    if os.path.exists(path_bow) == False:
        os.mkdir(path_bow)
    path_leftHand = os.path.join(result_folder, 'leftHand')
    if os.path.exists(path_leftHand) == False:
        os.mkdir(path_leftHand)
    path_rightHand = os.path.join(result_folder, 'rightHand')
    if os.path.exists(path_rightHand) == False:
        os.mkdir(path_rightHand)
    path_halfBody = os.path.join(result_folder, 'halfBody')
    if os.path.exists(path_halfBody) == False:
        os.mkdir(path_halfBody)
    path_qinThong = os.path.join(result_folder, 'qinThong')
    if os.path.exists(path_qinThong) == False:
        os.mkdir(path_qinThong)
    path_qinZhen = os.path.join(result_folder, 'qinZhen')
    if os.path.exists(path_qinZhen) == False:
        os.mkdir(path_qinZhen)

    videoInput = cv2.VideoCapture(video_path_from_react)
    # properties      = Util.getVideoProperties(videoInput)
    LABELS = ["body", "bow", "erhu"]
    segment_image = custom_segmentation()
    segment_image.inferConfig(num_classes=3, class_names=LABELS)
    segment_image.load_model("model/SegmentationModel/DatasetCombined_28012022/mask_rcnn_model.089-0.109433.h5")
    frame_number = 0
    frame_number_success = 0
    skipped_frame = 0
    selected_frame = 10
    max_frame = 50
    while videoInput.isOpened():
        if frame_number <= max_frame:
            # if skipped_frame == selected_frame :
            skipped_frame = 0
            success, frame = videoInput.read()
            if not success:
                break
            image = frame.copy()
            try:
                seg_mask, seg_output = segment_image.segmentFrame(frame.copy())
                segLeng = len(seg_mask['scores'])
                for i in range(segLeng):
                    mask = frame.copy()
                    id = seg_mask['class_ids'][i]
                    label = LABELS[int(id) - 1]
                    if mask.shape[0] == seg_mask['masks'].shape[0] and mask.shape[1] == seg_mask['masks'].shape[1]:
                        mask[seg_mask['masks'][:, :, i] == False] = (0, 0, 0)
                        if label == 'erhu':
                            image, erhu_line, QinZhenPart, QinThongPart = erhu_segment(image, mask)
                            cv2.imwrite(os.path.join(path_qinThong, "QinThong_" + str(frame_number) + ".jpg"),
                                        QinThongPart)
                            cv2.imwrite(os.path.join(path_qinZhen, "QinZhen_" + str(frame_number) + ".jpg"),
                                        QinZhenPart)
                        elif label == 'body':
                            image, leftHandPart, rightHandPart, rightArmPart, halfBodyPart = body_segment(image, mask)
                            cv2.imwrite(os.path.join(path_leftHand, "LeftHand_" + str(frame_number) + ".jpg"),
                                        leftHandPart)
                            cv2.imwrite(os.path.join(path_rightHand, "RightHand_" + str(frame_number) + ".jpg"),
                                        rightHandPart)
                            cv2.imwrite(os.path.join(path_halfBody, "HalfBody_" + str(frame_number) + ".jpg"),
                                        halfBodyPart)
                        elif label == 'bow':
                            image, bow_line, bow_cropped = bow_segment(image, mask)
                            cv2.imwrite(os.path.join(path_bow, "Bow_" + str(frame_number) + ".jpg"), bow_cropped)
                is_orthogonal = check_orthogonal(erhu_line[0][0], erhu_line[1][1], erhu_line[1][0], erhu_line[1][1], bow_line[0][0], bow_line[0][1], bow_line[1][0], bow_line[1][1])
                cv2.putText(image, '[E41][E42] Bow Erhu Orthogonal : ' + str(is_orthogonal), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.imshow("Video", image)
                # videoOut_1.write(image)
                frame_number += 1
                frame_number_success += 1
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                print("Frame ", str(frame_number), " processed!")
            except:
                print("Frame ", str(frame_number), " failed!")
                frame_number += 1
            # else:
            #     skipped_frame += 1
        else:
            frame_number = 0
            break
    videoInput.release()
    end_now = datetime.now()
    end_time = end_now.strftime("%H:%M:%S")
    print("Start Time =", start_now)
    print("End Time =", end_time)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Video path')
    argparser.add_argument('video_path',
                            type = str,
                            help ='video path')
    args = argparser.parse_args()
    _main_(args.video_path)