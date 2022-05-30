import re
import os
import cv2
import pdb
import glob
import pandas
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

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

det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# HR Net w48
pose_config = 'configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_256x192.py'
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
                [1,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16],
                [21,22,23,18,19,20],
                np.arange(40,23,-1), np.arange(50,40,-1),
                np.arange(51,55), np.arange(59,54,-1),
                [69,68,67,66,71,70], [63,62,61,60,65,64],
                np.arange(78,71,-1), np.arange(83,78,-1),
                [88,87,86,85,84,91,90,89],
                np.arange(113,134), np.arange(92,113)
                ]) - 1
assert(index_mirror.shape[0] == 133)

def merge_hm(hms_list):
    assert isinstance(hms_list, list)
    for hms in hms_list:
        hms[1,:,:,:] = torch.flip(hms[1,index_mirror,:,:], [2])

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

def csv2dict(anno_path, dataset_type):
    inputs_list = pandas.read_csv(anno_path)
    if dataset_type == 'train':
        broken_data = [2390]
        inputs_list.drop(broken_data, inplace=True)
    inputs_list = (inputs_list.to_dict()['id|folder|signer|annotation'].values())
    info_dict = dict()
    info_dict['prefix'] = anno_path.rsplit("/", 3)[0] + "/features/fullFrame-210x260px"
    print(f"Generate information dict from {anno_path}")
    for file_idx, file_info in tqdm(enumerate(inputs_list), total=len(inputs_list)):
        fileid, folder, signer, label = file_info.split("|")
        num_frames = len(glob.glob(f"{info_dict['prefix']}/{dataset_type}/{folder}"))
        info_dict[file_idx] = {
            'fileid': fileid,
            'folder': f"{dataset_type}/{folder}",
            'signer': signer,
            'label': label,
            'num_frames': num_frames,
            'original_info': file_info,
        }
        if DEBUG:
            break
    return info_dict


def generate_gt_stm(info, save_path):
    with open(save_path, "w") as f:
        for k, v in info.items():
            if not isinstance(k, int):
                continue
            f.writelines(f"{v['fileid']} 1 {v['signer']} 0.0 1.79769e+308 {v['label']}\n")


def sign_dict_update(total_dict, info):
    for k, v in info.items():
        if not isinstance(k, int):
            continue
        split_label = v['label'].split()
        for gloss in split_label:
            if gloss not in total_dict.keys():
                total_dict[gloss] = 1
            else:
                total_dict[gloss] += 1
    return total_dict


def resize_img(img_path, dsize='210x260px'):
    dsize = tuple(int(res) for res in re.findall("\d+", dsize))
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
    return img

def load_img(img_path):
    img = cv2.imread(img_path)
    return img


def resize_dataset(video_idx, dsize, info_dict):
    info = info_dict[video_idx]
    img_list = glob.glob(f"{info_dict['prefix']}/{info['folder']}")
    for img_path in img_list:
        rs_img = resize_img(img_path, dsize=dsize)
        rs_img_path = img_path.replace("210x260px", dsize)
        rs_img_dir = os.path.dirname(rs_img_path)
        if not os.path.exists(rs_img_dir):
            os.makedirs(rs_img_dir)
            cv2.imwrite(rs_img_path, rs_img)
        else:
            cv2.imwrite(rs_img_path, rs_img)


def visualize_dataset(video_idx, dsize, info_dict):
    info = info_dict[video_idx]
    img_list = glob.glob(f"{info_dict['prefix']}/{info['folder']}")
    for img_path in img_list:
        rs_img = resize_img(img_path, dsize=dsize)
        rs_img_path = img_path.replace("210x260px", dsize)
        rs_img_dir = os.path.dirname(rs_img_path)
        if not os.path.exists(rs_img_dir):
            os.makedirs(rs_img_dir)
            cv2.imwrite(rs_img_path, rs_img)
        else:
            cv2.imwrite(rs_img_path, rs_img)


def extract_keypoint(video_idx, info_dict, save_path, visualize=False):
    info = info_dict[video_idx]
    # print(info)
    # print(info_dict)
    img_list = glob.glob(f"{info_dict['prefix']}/{info['folder']}")

    keypoints = []

    for img_path in img_list:
        rs_img = load_img(img_path)
        rs_img_path = img_path.replace("210x260px", 'visualize')
        rs_img_dir = os.path.dirname(rs_img_path)

        # TODO visualize image output to 'rs_img'
        # rs_img =

        pose, visulization, keypoint = extract_keypoint_mmpose(rs_img, visualize)

        keypoints.append(keypoint)

        if visualize:
            if not os.path.exists(rs_img_dir):
                os.makedirs(rs_img_dir)
                cv2.imwrite(rs_img_path, visulization)
            else:
                cv2.imwrite(rs_img_path, visulization)

    # print(np.asarray(keypoints).shape)

    file_path = save_path + f"{info['fileid']}.npy"
    # print(save_path)
    # print(file_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        np.save(file_path, keypoints)
    else:
        np.save(file_path, keypoints)


def run_mp_cmd(processes, process_func, process_args):
    with Pool(processes) as p:
        outputs = list(tqdm(p.imap(process_func, process_args), total=len(process_args)))
    return outputs


def run_cmd(func, args):
    return func(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data process for Continuous Sign Language Recognition.')
    parser.add_argument('--dataset', type=str, default='phoenix2014',
                        help='save prefix')
    parser.add_argument('--dataset-root', type=str, default='dataset/phoenix2014/phoenix-2014-multisigner',
                        help='path to the dataset')
    parser.add_argument('--annotation-prefix', type=str, default='annotations/manual/{}.corpus.csv',
                        help='annotation prefix')
    parser.add_argument('--output-res', type=str, default='256x256px',
                        help='resize resolution for image sequence')
    parser.add_argument('--process-image', '-p', action='store_true',
                        help='resize image')
    parser.add_argument('--process-keypoint', '-k', action='store_true',
                        help='extract keypoint', default=False)
    parser.add_argument('--multiprocessing', '-m', action='store_true',
                        help='whether adopts multiprocessing to accelate the preprocess', default=False)

    args = parser.parse_args()
    mode = ["dev", "test", "train"]
    sign_dict = dict()
    if not os.path.exists(f"./{args.dataset}"):
        os.makedirs(f"./{args.dataset}")
    for md in mode:
        # generate information dict
        information = csv2dict(f"{args.dataset_root}/{args.annotation_prefix.format(md)}", dataset_type=md)
        # information = np.load(f"preprocess/phoenix2014/{md}_info.npy")
        np.save(f"./{args.dataset}/{md}_info.npy", information)
        # update the total gloss dict
        sign_dict_update(sign_dict, information)
        # generate groudtruth stm for evaluation
        # generate_gt_stm(information, f"./{args.dataset}/{args.dataset}-groundtruth-{md}.stm")
        # resize images
        video_index = np.arange(len(information) - 1)

        if args.process_image:
            print(f"Resize image to {args.output_res}")
            if args.multiprocessing:
                run_mp_cmd(10, partial(resize_dataset, dsize=args.output_res, info_dict=information), video_index)
            else:
                for idx in tqdm(video_index):
                    run_cmd(partial(resize_dataset, dsize=args.output_res, info_dict=information), idx)

        if args.process_keypoint:
            print(f"Process Keypoint")
            save_path_root = args.dataset_root + f'/features/keypoint/{md}/'
            if args.multiprocessing:
                run_mp_cmd(10, partial(extract_keypoint, info_dict=information, save_path=save_path_root), video_index)
            else:
                for idx in tqdm(video_index):
                    run_cmd(partial(extract_keypoint, info_dict=information, save_path=save_path_root), idx)
    sign_dict = sorted(sign_dict.items(), key=lambda d: d[0])
    save_dict = {}
    for idx, (key, value) in enumerate(sign_dict):
        save_dict[key] = [idx + 1, value]
    # np.save(f"./{args.dataset}/gloss_dict.npy", save_dict)
