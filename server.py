from flask import Flask, render_template, request, Response
from flask_socketio import SocketIO, emit
from datetime import datetime
from time import sleep
import cv2
import json
import base64
import eventlet
import numpy as np
from PIL import Image

eventlet.monkey_patch()

app = Flask(__name__)
app.config['SECRET_KEY'] = '78581099#lkjh'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# our gloabal worker
workerObject = None
PREDICTION_DIR = r"F:\Dataset\Erhu\Data Predicted\\"

class Worker(object):
    switch = False
    unit_of_work = 0

    def __init__(self, socketio):
        """
        assign socketio object to emit
        """
        self.socketio = socketio
        self.switch = True

    # def do_work(self):
    #     """
    #     do work and emit message
    #     """
    #     while self.switch:
    #         self.unit_of_work += 1
    #
    #         cap = cv2.VideoCapture('predict.mp4')
    #         frame_counter = 0
    #         while (cap.isOpened()):
    #             ret, img = cap.read()
    #             frame_counter += 1
    #             if ret and self.switch:
    #                 img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #                 frame = cv2.imencode('.jpg', img)[1].tobytes()
    #                 frame = base64.encodebytes(frame).decode("utf-8")
    #                 # message(frame, frame_counter)
    #                 data = {'image': frame, 'frame': frame_counter}
    #                 json_data = json.dumps(data)
    #                 self.socketio.emit('image', json_data, namespace="/work")
    #                 # socketio.sleep(0.05)
    #                 eventlet.sleep(0.05)
    #             else:
    #                 break

    def do_work(self):
        """
        do work and emit message
        """
        while self.switch:
            self.unit_of_work += 1

            file = np.load('output.npz', allow_pickle=True)
            file = file['arr_0']

            # print(len(file))
            # break
            # print(file[0][0])
            # print(file[0][1])
            # break
            frame_counter = 0
            for frame in file:
                frame_counter += 1
                # img = Image.fromarray(frame[1], 'RGB').convert('RGB')
                img = frame[1]

                if self.switch:
                    img = cv2.imencode('.jpg', img)[1].tobytes()
                    img = base64.encodebytes(img).decode("utf-8")
                    # message(frame, frame_counter)
                    data = {'image': img, 'frame': frame_counter, 'data': frame[0]}
                    json_data = json.dumps(data)
                    self.socketio.emit('image', json_data, namespace="/work")
                    # socketio.sleep(0.05)
                    eventlet.sleep(0.1)
                else:
                    self.switch = False
                    break

            eventlet.sleep(1)
            self.switch = False

            # cap = cv2.VideoCapture('predict.mp4')
            # frame_counter = 0
            # while (cap.isOpened()):
            #     ret, img = cap.read()
            #     frame_counter += 1
            #     if ret and self.switch:
            #         img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            #         frame = cv2.imencode('.jpg', img)[1].tobytes()
            #         frame = base64.encodebytes(frame).decode("utf-8")
            #         # message(frame, frame_counter)
            #         data = {'image': frame, 'frame': frame_counter}
            #         json_data = json.dumps(data)
            #         self.socketio.emit('image', json_data, namespace="/work")
            #         # socketio.sleep(0.05)
            #         eventlet.sleep(0.05)
            #     else:
            #         break

            # important to use eventlet's sleep method
            # eventlet.sleep(1)

    def stop(self):
        """
        stop the loop
        """
        self.switch = False


@socketio.on('connect', namespace='/work')
def connect():
    """
    connect
    """
    global worker
    worker = Worker(socketio)
    emit("re_connect", {"msg": "connected"})


@socketio.on('play', namespace='/work')
def start_work():
    print('start worker video')
    """
    trigger background thread
    """
    emit("update", {"msg": "starting worker"})
    # notice that the method is not called - don't put braces after method name
    socketio.start_background_task(target=worker.do_work)


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

@app.route("/send-video", methods=['POST'])
def send_video():
    video = request.files.get('video')

    now = datetime.now()  # current date and time

    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    print("date and time:", date_time)

    video.filename = date_time + ".mp4"

    # Saved Video Path
    filename = PREDICTION_DIR + video.filename

    print(filename)

    video.save(filename)

if __name__ == "__main__":
    socketio.run(app, debug=True, host='140.115.51.243', port=5000)
