import os
import subprocess
from datetime import datetime
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from ErhuPrediction3DCNNLSTM_class import main_predict

app = Flask(__name__)
CORS(app)
@app.route('/')
def hello_words():
    print('Hello Words')
    return 'Hello Words'

@app.route('/submit_video', methods=['POST', 'GET'])
def upload_video():
    if request.method == 'POST':
        # video = request.form['video']
        video = request.files['video']
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H%M%S")
        video_save_path = os.path.join("video", dt_string)
        video_filename  = "result_"+dt_string+".mp4"
        video_saved_file = os.path.join(video_save_path, video_filename)
        if os.path.exists(video_save_path) == False:
            os.mkdir(video_save_path)
        video.save(video_saved_file)
        print(video_saved_file)
        main_predict(video_saved_file)
        # subprocess.run("python segmentation_process.py "+video_saved_file, shell=True)
        # os.system("segmentation_process.py "+video_saved_file)
        # exec(open('segmentation_process.py '+video_saved_file).read())
        # segmentation_process(video_saved_file)

        # video.flush()
        # video.close()
        # try :
        # stream = ffmpeg.input(video)
        # stream = ffmpeg.output(stream, 'G:/Aditya/MINELab Project/ErhuBackendReact/result.mp4')
        # ffmpeg.run(stream)
        # except :
        #     print('Error....!')
        print('Submited POST')
        return 'Success POST'
    else:
        # video = request.args.get('video')
        video = request.files['video'].stream.read()
        # try:
        # stream = ffmpeg.input(video)
        # stream = ffmpeg.output(stream, 'G:/Aditya/MINELab Project/ErhuBackendReact/result.mp4')
        # ffmpeg.run(stream)
        # except:
        #     print('Error....!')
        print('Submited GET')
        return 'Success GET'


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port='3000')
    app.env = 'development'