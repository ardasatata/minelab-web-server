import os.path

from flask import Flask, request, jsonify, send_file
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

from natsort import os_sorted

import shutil

# import random

sys.path.append(os.path.abspath(os.path.join('..', 'classifier')))

from classifier.timeout import timeout
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

# Directory List
PREDICTION_DIR = r"./predict/"
UPLOAD_DIR = r"./upload/"
PROCESSED_DIR = r"./processed/"

UPLOAD_DIR_SEND_FILE = r"/home/minelab/dev/erhu-project/upload/"
PREDICT_DIR_SEND_FILE = r"/home/minelab/dev/erhu-project/predict/"
PROCESSED_DIR_SEND_FILE = r"/home/minelab/dev/erhu-project/processed/"
DELETED_DIR_SEND_FILE = r"/home/minelab/dev/erhu-project/deleted/"


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


@app.route("/send-video", methods=['POST'])
@cross_origin()
def send_video():
    video = request.files.get('video')
    print('received..', video.filename)

    now = datetime.now()  # current date and time
    date_time = now.strftime("%d_%m_%Y_%A(%H:%M:%S)")

    # Video Metadata
    filename = f"{date_time}.webm"
    processed_filename = f"{date_time}.mp4"

    video.filename = filename

    # Saved Video Path
    filename = video.filename

    value = {
        "ok": True,
        "filename": filename
    }

    print("saving...", filename)
    video.save(UPLOAD_DIR + filename)

    try:
        # run classifier & blurring sub-process
        print("blurring...", filename)
        subprocess.Popen(
            ['python', './server/FaceMosaicMediaPipe.py', UPLOAD_DIR + filename, PROCESSED_DIR + processed_filename])

        print("processing...", filename)
        subprocess.Popen(['python', './classifier/classifier.py', UPLOAD_DIR + filename])

    except subprocess.CalledProcessError as e:
        print("Unexpected error:", e)
        value = {
            "ok": False,
            "filename": filename
        }
        print("error uploading file")
        return value
    return value


@app.route('/delete', methods=['GET'])
def delete_original():
    filename = request.args.get('filename')
    file_path = UPLOAD_DIR_SEND_FILE + filename
    deleted_file_path = DELETED_DIR_SEND_FILE + filename
    print('deleting...', file_path)

    if isfile(file_path):
        try:
            shutil.move(file_path, deleted_file_path)
        except PermissionError:
            print('error delete_original')
            shutil.os.system('sudo chown $USER "{}"'.format(file_path))
            # try again
            try:
                shutil.move(file_path, deleted_file_path)
            except:
                print('Giving up on'.format(file_path))

        value = {
            "ok": True,
            "filename": filename,
            "message": 'File deleted!'
        }
        print('OK Deleted')
        return value
    else:
        value = {
            "ok": False,
            "filename": filename,
            "message": 'File not found!'
        }
        print('Error Deleted')
        print('File not found!')
        return value


@app.route('/download-original', methods=['GET'])
def download_original():
    filename = request.args.get('filename')
    file_path = PROCESSED_DIR_SEND_FILE + filename
    print('downloading...', file_path)

    if isfile(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        value = {
            "ok": False,
            "filename": filename,
            "message": 'File not found!'
        }
        return value


@app.route('/download-predict', methods=['GET'])
def download_predict():
    filename = request.args.get('filename')
    file_path = PREDICT_DIR_SEND_FILE + filename
    print('downloading...', file_path)

    if isfile(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        value = {
            "ok": False,
            "filename": filename,
            "message": 'File not found!'
        }
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

    # try:
    #     isError, img, list_err = check_player_img_postition(img)
    #     # signal.alarm(1)
    # except TimeoutException as e:
    #     print("Timed out!")

    isError, img, list_err = check_player_img_postition(img)

    # simulate error
    # isError = bool(random.getrandbits(1))
    message = ""

    # print(isError)

    # if isError:
    #     message = 'Error occurred'

    while True:
        cv2.imshow('webcam', img)
        # cv2.waitKey(900)
        cv2.destroyAllWindows()
        break
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    data = {
        "ok": isError,
        "message": message,
        "errors": list_err
    }

    base64_img = cv2.imencode('.jpg', img)[1].tobytes()
    base64_img = base64.encodebytes(base64_img).decode("utf-8")
    data = {'image': base64_img, 'data': data}
    json_data = json.dumps(data)
    emit('image', json_data, namespace="/stream-checking")


@app.route('/predict-list')
def file_path():
    print('getting all predicted files name')
    onlyfiles = [f[:-4] for f in os_sorted(listdir(PREDICTION_DIR)) if
                 isfile(join(PREDICTION_DIR, f)) and f[-4:] == ".npz"]

    print(onlyfiles)
    value = {
        "filepath": onlyfiles
    }
    return jsonify(value)


@app.route('/file-list')
def file_list():
    # print('getting all predicted files name')
    # onlyfiles = [f[:-4] for f in os_sorted(listdir(UPLOAD_DIR))]

    list = []

    for f in os_sorted(listdir(UPLOAD_DIR)):
        list.append({
            "filename": f[:-5],
            "original": f,
            "processed": f"{f[:-5]}.mp4",
            "isProcessing": False if isfile(join(PREDICTION_DIR, f"{f[:-5]}.npz")) else True,
        })

    # print(list)

    value = {
        "ok": True,
        "filepath": list
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


if __name__ == "__main__":
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, certfile="server/140_115_51_243.chained.crt",
                 keyfile="server/140_115_51_243.key")
    # socketio.run(app, debug=True, host='0.0.0.0', port=5000)
