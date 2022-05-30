import numpy as np

# Check Pytorch installation
import torch, torchvision

print('torch version:', torch.__version__, torch.cuda.is_available())
print('torchvision version:', torchvision.__version__)

# Check MMPose installation
import mmpose

print('mmpose version:', mmpose.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

print('cuda version:', get_compiling_cuda_version())
print('compiler information:', get_compiler_version())

import cv2
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)

from mmdet.apis import inference_detector, init_detector

local_runtime = False

# # Original Config
# pose_config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
# pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'

det_config = '/home/minelab/dev/VAC_CSLR/preprocess/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# HR Net w48
pose_config = '/home/minelab/dev/VAC_CSLR/preprocess/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_256x192-643e18cb_20200922.pth'

# Higher HRNet
# pose_config = 'configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/vipnas_res50_coco_wholebody_256x192.py'
# pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_res50_wholebody_256x192-49e1c3a4_20211112.pth'

# det_config = 'demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_coco.py'
# det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth'

# initialize pose model
pose_model = init_pose_model(pose_config, pose_checkpoint)
# initialize detector
det_model = init_detector(det_config, det_checkpoint)

DEBUG = False

index_mirror = np.concatenate([
    [1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16],
    [21, 22, 23, 18, 19, 20],
    np.arange(40, 23, -1), np.arange(50, 40, -1),
    np.arange(51, 55), np.arange(59, 54, -1),
    [69, 68, 67, 66, 71, 70], [63, 62, 61, 60, 65, 64],
    np.arange(78, 71, -1), np.arange(83, 78, -1),
    [88, 87, 86, 85, 84, 91, 90, 89],
    np.arange(113, 134), np.arange(92, 113)
]) - 1
assert (index_mirror.shape[0] == 133)


def merge_hm(hms_list):
    assert isinstance(hms_list, list)
    for hms in hms_list:
        hms[1, :, :, :] = torch.flip(hms[1, index_mirror, :, :], [2])


def extract_keypoint_mmpose(img, visualize=False):
    # inference detection
    mmdet_results = inference_detector(det_model, img)

    # extract person (COCO_ID=1) bounding boxes from the detection results
    person_results = process_mmdet_results(mmdet_results, cat_id=1)

    # inference pose
    pose_results, returned_outputs = inference_top_down_pose_model(pose_model,
                                                                   img,
                                                                   person_results,
                                                                   bbox_thr=0.3,
                                                                   format='xyxy',
                                                                   dataset=pose_model.cfg.data.test.type)

    vis_result = None

    if visualize:
        # show pose estimation results
        vis_result = vis_pose_result(pose_model,
                                     img,
                                     pose_results,
                                     dataset=pose_model.cfg.data.test.type,
                                     show=False)
        # reduce image size
        # vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)

    keypoints = pose_results[0].get('keypoints')
    bbox = pose_results[0].get('bbox')

    # print(keypoints)
    # print(np.asarray(keypoints).shape)
    #
    # print(bbox)
    #
    #
    # exit()
    return pose_results, vis_result, keypoints


def extract_keypoint(filename="92_wisnu (online-video-cutter.com).mp4"):
    keypoints = []

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(filename)

    cap.set(cv2.CAP_PROP_FPS, 30)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'keypoint_{filename}', fourcc, 30,
                          (512, 512))

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # # Display the resulting frame
            # cv2.imshow('Frame', frame)

            frame = cv2.resize(frame, (512, 512))

            pose, visulization, keypoint = extract_keypoint_mmpose(frame, True)

            # Display the resulting frame
            # cv2.imshow('Visualization', visulization)

            out.write(visulization)
            #
            keypoints.append(keypoint)

            # # Press Q on keyboard to  exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    file_path = f"{filename}.npy"
    np.save(file_path, keypoints)


if __name__ == '__main__':
    extract_keypoint("92_wisnu (online-video-cutter.com).mp4")
