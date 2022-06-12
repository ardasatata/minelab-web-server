import sys
import subprocess
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

path1 = sys.argv[1] # UPLOAD_DIR + filename_temp
path2 = sys.argv[2] # UPLOAD_DIR + filename
path3 = sys.argv[3] # SLICED_DIR + filename
path4 = sys.argv[4] # PROCESSED_DIR + processed_filename
path5 = sys.argv[5] # PREDICT_DIR_SEND_FILE + processed_blurred
path6 = sys.argv[6] # PREDICT_DIR_SEND_FILE + stream_file
filename = sys.argv[7] # filename

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def with_opencv(filename):
    import cv2
    video = cv2.VideoCapture(filename)

    duration = video.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    return duration, frame_count


subprocess.run(["ffmpeg", "-i", path1, '-preset', 'superfast', path2])
# subprocess.run(["ffmpeg", "-an", "-i", path1, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-profile:v', 'baseline', '-level', '3', path2])

video_length = get_length(path2)

print('vid_length :', video_length)
print('vid_length sliced :', video_length - 2.0)
print('vid_length :', with_opencv(path2))

subprocess.run(["rm", "-rf", path1])

ffmpeg_extract_subclip(path2, 0, video_length - 2.0, targetname=path3)
print('vid_length sliced:', with_opencv(path3))

# run classifier & blurring sub-process
print("blurring...", filename)
subprocess.Popen(
    ['python', './server/FaceMosaicMediaPipe.py', path3, path4])

print("processing...", filename)
subprocess.Popen(
    ['python', './classifier/classifier.py', path3, path5, path6])

