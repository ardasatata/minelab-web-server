import os.path

from flask import Flask, render_template, request, Response, jsonify
from flask_socketio import SocketIO, emit
from datetime import datetime
import cv2
import json
import base64
import eventlet
import numpy as np

from os import listdir
from os.path import isfile, join

from flask_cors import CORS, cross_origin

import subprocess

import sys

from util import readb64

import random

sys.path.append(os.path.abspath(os.path.join('..', 'classifier')))

from classifier.checking_tool import check_player_img_postition

# from classifier.ErhuPrediction3DCNNLSTM_class import check_player_postition

eventlet.monkey_patch()

app = Flask(__name__)
cors = CORS(app)

app.config['SECRET_KEY'] = '78581099#lkjh'
app.config['CORS_HEADERS'] = 'Content-Type'

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# our global worker
workerObject = None
PREDICTION_DIR = r"./predict/"
UPLOAD_DIR = r"./upload/"

import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

from multiprocessing import Process
from time import sleep

def f(time):
    sleep(time)


def run_with_limited_time(func, args, kwargs, time):
    """Runs a function with time limit

    :param func: The function to run
    :param args: The functions args, given as tuple
    :param kwargs: The functions keywords, given as dict
    :param time: The time limit in seconds
    :return: True if the function ended successfully. False if it was terminated.
    """
    p = Process(target=func, args=args, kwargs=kwargs)
    p.start()
    p.join(time)
    if p.is_alive():
        p.terminate()
        return False

    return True

class Worker(object):
    switch = False
    unit_of_work = 0

    def __init__(self, socketio):
        """
        assign socketio object to emit
        """
        self.socketio = socketio
        self.switch = True

    def do_work(self, filename='output'):
        """
        do work and emit message
        """
        while self.switch:
            self.unit_of_work += 1

            file = np.load(f"{PREDICTION_DIR}{filename}.npz", allow_pickle=True)
            file = file['arr_0']
            #
            # video = skvideo.io.vread(f"{PREDICTION_DIR}{filename}.mp4")
            print(f"{PREDICTION_DIR}{filename}.mp4")

            vid = cv2.VideoCapture(f"{PREDICTION_DIR}{filename}.mp4")

            video = []
            check = True
            i = 0

            while check:
                check, arr = vid.read()
                # if not i % 20:  # This line is if you want to subsample your video
                #     # (i.e. keep one frame every 20)
                video.append(arr)
                i += 1

            video = np.array(video)

            frame_counter = 0
            for index in range(file.shape[0]):
                # print(video[index])
                frame_counter += 1
                # img = Image.fromarray(frame[1], 'RGB').convert('RGB')
                # img = frame[1]
                img = video[index]

                if self.switch:
                    img = cv2.imencode('.jpg', img)[1].tobytes()
                    img = base64.encodebytes(img).decode("utf-8")
                    # message(frame, frame_counter)
                    data = {'image': img, 'frame': frame_counter, 'data': file[index][0].tolist()}
                    json_data = json.dumps(data)
                    self.socketio.emit('image', json_data, namespace="/work")
                    # socketio.sleep(0.05)
                    eventlet.sleep(0.075)
                else:
                    self.switch = False
                    break

            eventlet.sleep(1)
            self.switch = False

    def stop(self):
        """
        stop the loop
        """
        self.switch = False


# Getting arguments from a POST form
@app.route("/send-video", methods=['POST'])
@cross_origin()
def send_video():
    video = request.files.get('video')

    now = datetime.now()  # current date and time

    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    print("date and time:", date_time)

    original_filename = video.filename[:-4]

    video.filename = f"{original_filename}__({date_time}).mp4"

    # Saved Video Path
    # filename = UPLOAD_DIR + video.filename
    filename = video.filename

    value = {
        "ok": True,
        "fileName": date_time
    }

    print("saving...", filename)
    video.save(UPLOAD_DIR + filename)

    try:
        # # run checker sub-process
        # print("chekcing...", filename)
        # subprocess.run(['python', './classifier/checker.py', UPLOAD_DIR + filename])

        # run classifier sub-process
        print("processing...", filename)
        subprocess.Popen(['python', './classifier/classifier.py', UPLOAD_DIR + filename], stdout=subprocess.DEVNULL,
                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("Unexpected error:", e)
        value = {
            "ok": False,
            "fileName": filename
        }
        print("error uploading file")
        return value

    return value


@socketio.on('connect', namespace='/stream-checking')
def connect():
    """
    connect
    """
    print('connect, /stream-checking')
    emit("re_connect", {"msg": "connected"})


@socketio.on('webcam-stream', namespace='/stream-checking')
def start_work(data):
    img = readb64(data)
    # print('receive webcam stream ', img)

    """
    frame prediction goes here ~~~
    """

    try:
        with time_limit(1):
            isError, img, list_err = check_player_img_postition(img)
    except TimeoutException as e:
        print("Timed out!")

    # simulate error
    # isError = bool(random.getrandbits(1))
    message = "All Ok!"

    # print(isError)

    if isError:
        message = 'Error occurred'

    while True:
        cv2.imshow('webcam', img)
        # cv2.waitKey(900)
        cv2.destroyAllWindows()
        break
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    data = {
        "ok": not isError,
        "message": message
    }

    base64_img = cv2.imencode('.jpg', img)[1].tobytes()
    base64_img = base64.encodebytes(base64_img).decode("utf-8")
    data = {'image': base64_img, 'data': data}
    json_data = json.dumps(data)
    emit('image', json_data, namespace="/stream-checking")


@app.route('/predict-list')
def file_path():
    print('getting all predicted files name')
    onlyfiles = [f[:-4] for f in listdir(PREDICTION_DIR) if isfile(join(PREDICTION_DIR, f)) and f[-4:] == ".npz"]

    print(onlyfiles)
    value = {
        "filepath": onlyfiles
    }
    return jsonify(value)


@socketio.on('connect', namespace='/work')
def connect():
    """
    connect
    """
    global worker
    worker = Worker(socketio)
    emit("re_connect", {"msg": "connected"})


@socketio.on('play', namespace='/work')
def start_work(data):
    print('start worker video ', data)
    """
    trigger background thread
    """
    emit("update", {"msg": "starting worker"})
    # notice that the method is not called - don't put braces after method name
    # socketio.start_background_task(target=worker.do_work('output_professor.npz'))
    socketio.start_background_task(target=worker.do_work(data))


@socketio.on('stop', namespace='/work')
def stop_work():
    print('stop worker trigger')
    """
    trigger background thread
    """
    worker.stop()
    emit("update", {"msg": "worker has been stoppped"})


@socketio.on('disconnect', namespace='/work')
def stop_work():
    print('disconnect')
    """
    trigger background thread
    """
    worker.stop()
    emit("update", {"msg": "worker has been stoppped"})


# @app.route("/send-video", methods=['POST'])
# def send_video():
#     video = request.files.get('video')
#
#     now = datetime.now()  # current date and time
#
#     date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
#     print("date and time:", date_time)
#
#     video.filename = date_time + ".mp4"
#
#     # Saved Video Path
#     filename = PREDICTION_DIR + video.filename
#
#     print(filename)
#
#     video.save(filename)


if __name__ == "__main__":
    socketio.run(app, debug=True, host='140.115.51.243', port=5000)
