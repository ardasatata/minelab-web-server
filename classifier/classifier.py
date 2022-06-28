import sys
# from ErhuPrediction3DCNNLSTM_class import main_predict
from ErhuPredictionYOLO_GCN_class import main_predict
import mediapipe as mp
import subprocess


inputVideo = sys.argv[1]
outPredictName = sys.argv[2]
streamFileName = sys.argv[3]
analyzedBlurred = sys.argv[4]

# main_predict('/home/minelab/dev/erhu-project/upload/04_19_2022_02_38_16_04_15_2022_10_04_25_01.mp4')
result, result_error_msg = main_predict(inputVideo)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

out_filename = outPredictName

print("blurring...", out_filename)
subprocess.call(
    ['python', './server/FaceMosaicMediaPipe.py', result, out_filename])
print("blurring prediction done...", out_filename)

print("blurring analyzed message...", analyzedBlurred)
subprocess.call(
    ['python', './server/FaceMosaicMediaPipe.py', result_error_msg, analyzedBlurred])
print("blurring prediction done...", analyzedBlurred)

subprocess.run(["rm", "-rf", result_error_msg])

filename_transform = out_filename.replace('(', '\(')
filename_transform = filename_transform.replace(':', '\:')
filename_transform = filename_transform.replace(')', '\)')

filename_transform_out = streamFileName.replace('(', '\(')
filename_transform_out = filename_transform_out.replace(':', '\:')
filename_transform_out = filename_transform_out.replace(')', '\)')

subprocess.call(
    ["ffmpeg", "-an", "-i", out_filename, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-profile:v', 'baseline', '-level',
     '3', streamFileName])

print("stream file done...", streamFileName)
