import math
import numpy as np
import cv2
from termcolor import colored
import os


CAMERA = [
    {
        'ID' : 1,           # First camera ID for RGB camera
        'WIDTH' : 640,
        'HEIGHT' : 480,
        'FPS' : 30
    },
    {
        'ID' : 0,
        'WIDTH' : 640,
        'HEIGHT' : 480,
        'FPS' : 30,
        'DEPTH' : True      # Depth camera should be True
    }
]

IMG_SHOW_MAX_WIDTH = 800

DATA_BASE_DIRECTORY = "data"
DATA_NEW_VIDEO_DIRECTORY = "new_video"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def getVideoProperties(video) :
    properties = {}
    properties['FPS'] = round(video.get(cv2.CAP_PROP_FPS))
    properties['Height'] = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    properties['Width'] = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    properties['Count'] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    return properties

def hhmmss(ms) :
    s = round(ms / 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return ("%d:%02d:%02d" % (h,m,s)) if h else ("%02d:%02d" % (m,s))

def distance2D(p1, p2) :
    distance = math.sqrt(((int(p1[0]) - int(p2[0])) ** 2) + ((int(p1[1]) - int(p2[1])) ** 2))
    return distance

def fixPoint(approx, flip = False) :
    if flip :
        approx = np.flip(approx, 0)
    tmp = approx.copy()

    tmp[0] = approx[3]
    tmp[1] = approx[0]
    tmp[2] = approx[1]
    tmp[3] = approx[2]

    return tmp

def isRectangle(approx, range = 20) :
    d_1 = math.sqrt(((approx[0][0][0] - approx[1][0][0]) ** 2) + ((approx[0][0][1] - approx[1][0][1]) ** 2))
    if d_1 < range :
        return False

    d_2 = math.sqrt(((approx[2][0][0] - approx[3][0][0]) ** 2) + ((approx[2][0][1] - approx[3][0][1]) ** 2))
    if d_2 < range :
        return False

    return True

def default(obj) :
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

def PrintPercent(c, t) :
    percent = math.floor(c / t * 100)

    strSize = 30
    str = ""
    for i in range(strSize) :
        p = math.floor(i / strSize * 100)
        if p < percent :
            str += "="
        else :
            str += "_"

    print("{} {}{}%{} ({}/{})".format(str, bcolors.WARNING, percent, bcolors.ENDC, c, t), end="\r")
