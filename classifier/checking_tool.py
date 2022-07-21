import cv2
import mediapipe as mp

from classifier.timeout import timeout

import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    # signal.setitimer()
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# @timeout(1)
def check_player_img_postition(img):
    image = img.copy()

    mp_holistic_checker = mp.solutions.holistic

    try:
        with time_limit(1):
            with mp_holistic_checker.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3, smooth_landmarks=True, static_image_mode=True) as holistic:
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image_height, image_width, _ = image.shape
                image.flags.writeable = True
                try:
                    x_LK = round(
                        results.pose_landmarks.landmark[mp_holistic_checker.PoseLandmark.LEFT_KNEE].x * image_width)
                    y_LK = round(
                        results.pose_landmarks.landmark[mp_holistic_checker.PoseLandmark.LEFT_KNEE].y * image_height)
                    x_RK = round(
                        results.pose_landmarks.landmark[mp_holistic_checker.PoseLandmark.RIGHT_KNEE].x * image_width)
                    y_RK = round(
                        results.pose_landmarks.landmark[mp_holistic_checker.PoseLandmark.RIGHT_KNEE].y * image_height)
                    image = cv2.circle(image, (x_LK, y_LK), 5, (255, 0, 0), thickness=-1)
                    image = cv2.circle(image, (x_RK, y_LK), 5, (255, 0, 0), thickness=-1)
                    if (y_RK < image_height) and (y_LK < image_height):
                        knee_detected = True
                    else:
                        knee_detected = False
                except:
                    knee_detected = False
                try:
                    x_lf_shoulder = round(
                        results.pose_landmarks.landmark[mp_holistic_checker.PoseLandmark.LEFT_SHOULDER].x * image_width)
                    y_lf_shoulder = round(results.pose_landmarks.landmark[
                                              mp_holistic_checker.PoseLandmark.LEFT_SHOULDER].y * image_height)
                    x_rh_shoulder = round(results.pose_landmarks.landmark[
                                              mp_holistic_checker.PoseLandmark.RIGHT_SHOULDER].x * image_width)
                    y_hip = round(
                        results.pose_landmarks.landmark[mp_holistic_checker.PoseLandmark.LEFT_HIP].y * image_height)
                    y_hip_shoulder_diff = abs(y_hip - y_lf_shoulder)
                    x_middle_shoulder = x_lf_shoulder - abs(x_rh_shoulder - x_lf_shoulder) // 2
                    y_middle_shoulder = y_lf_shoulder
                    left_area = (image_width // 2) - (abs(x_rh_shoulder - x_lf_shoulder) // 2)
                    right_area = (image_width // 2) + (abs(x_rh_shoulder - x_lf_shoulder) // 2)
                    y_top_erhu = y_lf_shoulder - y_hip_shoulder_diff
                    x_top_erhu = x_lf_shoulder
                    image = cv2.circle(image, (x_top_erhu, y_top_erhu), 5, (255, 0, 0), thickness=-1)
                    image = cv2.circle(image, (x_top_erhu, y_top_erhu), 5, (255, 0, 0), thickness=-1)
                    image = cv2.line(image, (x_middle_shoulder, y_middle_shoulder),
                                     (x_middle_shoulder, y_middle_shoulder + 100), (255, 0, 0), thickness=3)

                    if (y_top_erhu > 0):
                        top_erhu_detected = True
                    else:
                        top_erhu_detected = False
                    if (x_middle_shoulder > left_area and x_middle_shoulder < right_area):
                        center_body = True
                    else:
                        center_body = False
                except:
                    top_erhu_detected = False
                    center_body = False

    except TimeoutException as e:
        print("Timed out!")

    ret_val = [knee_detected, top_erhu_detected, center_body]
    # return ret_val
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if (knee_detected == True and top_erhu_detected == True and center_body == True):
        return True, image, ret_val
    else:
        return False, image, ret_val
