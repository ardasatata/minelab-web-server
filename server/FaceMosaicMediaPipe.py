import cv2
import mediapipe as mp
import sys

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

input = sys.argv[1]
output = sys.argv[2]

videoInput = cv2.VideoCapture(input)
face_detection = mp_face_detection.FaceDetection(model_selection = 1, min_detection_confidence = 0.4)

videoOut = None

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
FPS = 30.0

preview = False
out_filename = output

while videoInput.isOpened():
    success, frame = videoInput.read()
    if not success :
        break

    image = frame.copy()

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    if results.detections:
        for detection in results.detections:
            try :
                box = detection.location_data.relative_bounding_box
                box = int(image.shape[1] * box.xmin * 1.025), int(image.shape[0] * box.ymin * 0.95), int(image.shape[1] * box.width * 0.955), int(image.shape[0] * box.height)

                face = image[box[1]: box[1] + box[3], box[0]: box[0] + box[2]]

                tmp = cv2.resize(face, (8, 8), interpolation=cv2.INTER_NEAREST)
                face = cv2.resize(tmp, (face.shape[1], face.shape[0]), interpolation=cv2.INTER_NEAREST)

                #image[box[1]: box[1] + box[3], box[0]: box[0] + box[2]] = face
                halfHeight = int(face.shape[0]/2)
                width3 = face.shape[1] / 5
                widthDif = face.shape[1] - width3
                image[box[1]: box[1] + halfHeight, box[0]: box[0] + box[2]] = face[:halfHeight,:]

                for i in range(halfHeight) :
                    width = int(widthDif + width3 * (1 - i / halfHeight))
                    start = face.shape[1] - width
                    image[box[1]: box[1] + halfHeight + i, box[0] + start: box[0] + width] = face[:halfHeight + i, start : width]

                # if results.detections:
                #     for detection in results.detections:
                #         mp_drawing.draw_detection(image, detection)

            except:
                ''
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if videoOut is None :
        videoOut = cv2.VideoWriter(out_filename, fourcc, FPS, (image.shape[1], image.shape[0]))

    if videoOut is not None :
        videoOut.write(image)

    if preview:
        cv2.imshow("Outputs", image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

videoInput.release()
print("blurring done...", out_filename)
