import sys
from ErhuPrediction3DCNNLSTM_class import main_predict

import cv2
import mediapipe as mp

import subprocess

inputVideo = sys.argv[1]
outPredictName = sys.argv[2]
streamFileName = sys.argv[3]

# main_predict('/home/minelab/dev/erhu-project/upload/04_19_2022_02_38_16_04_15_2022_10_04_25_01.mp4')
result = main_predict(inputVideo)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

videoInput = cv2.VideoCapture(result)
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)

videoOut = None

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
FPS = 30.0

preview = False
out_filename = outPredictName

while videoInput.isOpened():
    success, frame = videoInput.read()
    if not success:
        break

    image = frame.copy()

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    if results.detections:
        for detection in results.detections:
            try:
                box = detection.location_data.relative_bounding_box
                box = int(image.shape[1] * box.xmin * 1.025), int(image.shape[0] * box.ymin * 0.95), int(
                    image.shape[1] * box.width * 0.955), int(image.shape[0] * box.height)

                face = image[box[1]: box[1] + box[3], box[0]: box[0] + box[2]]

                tmp = cv2.resize(face, (8, 8), interpolation=cv2.INTER_NEAREST)
                face = cv2.resize(tmp, (face.shape[1], face.shape[0]), interpolation=cv2.INTER_NEAREST)

                # image[box[1]: box[1] + box[3], box[0]: box[0] + box[2]] = face
                halfHeight = int(face.shape[0] / 2)
                width3 = face.shape[1] / 5
                widthDif = face.shape[1] - width3
                image[box[1]: box[1] + halfHeight, box[0]: box[0] + box[2]] = face[:halfHeight, :]

                for i in range(halfHeight):
                    width = int(widthDif + width3 * (1 - i / halfHeight))
                    start = face.shape[1] - width
                    image[box[1]: box[1] + halfHeight + i, box[0] + start: box[0] + width] = face[:halfHeight + i,
                                                                                             start: width]

                # if results.detections:
                #     for detection in results.detections:
                #         mp_drawing.draw_detection(image, detection)

            except:
                ''
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if videoOut is None:
        videoOut = cv2.VideoWriter(out_filename, fourcc, FPS, (image.shape[1], image.shape[0]))

    if videoOut is not None:
        videoOut.write(image)

    if preview:
        cv2.imshow("Outputs", image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

videoInput.release()
print("blurring prediction done...", out_filename)

subprocess.run(
    ["ffmpeg", "-an", "-i", result, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-profile:v', 'baseline', '-level',
     '3', streamFileName])

print("stream file done...", streamFileName)
