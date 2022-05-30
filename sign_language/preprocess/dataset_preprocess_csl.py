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

sentences = {0: ['他的', '同学', '是', '警察'], 1: ['他', '妈妈', '的', '同学', '是', '公务', '员'], 2: ['我的', '爸爸', '是', '商人'],
             3: ['他', '哥哥', '的', '目标', '是', '解放', '军'], 4: ['他', '姐姐', '的', '目标', '是', '模特'],
             5: ['我', '朋友', '的', '祖父', '是', '工人'], 6: ['我', '同学', '的', '妈妈', '是', '保姆'],
             7: ['我', '同学', '的', '妹妹', '是', '护士'], 8: ['我的', '妻子', '是', '教师'], 9: ['我的', '丈夫', '是', '导演'],
             10: ['他', '哥哥', '的', '朋友', '是', '演员'], 11: ['他的', '女朋友', '是', '护士'],
             12: ['她', '丈夫', '的', '朋友', '是', '教练'], 13: ['他', '哥哥', '的', '同学', '是', '医生'],
             14: ['他', '妹妹', '的', '同学', '是', '律师'], 15: ['他的', '邻居', '是', '残疾', '人'], 16: ['他的', '外祖母', '是', '园丁'],
             17: ['他的', '祖父', '是', '炊事员'], 18: ['我', '表哥', '的', '邻居', '是', '记者'], 19: ['他的', '丈夫', '是', '警察'],
             20: ['我', '公公', '是', '牧民'], 21: ['我', '婆婆', '是', '保育', '员'], 22: ['他', '儿子', '是', '弱', '智', '人'],
             23: ['我', '嫂嫂', '是', '画家'], 24: ['你', '祖父', '是', '知识', '分子'], 25: ['你', '外祖父', '是', '猎', '手'],
             26: ['他', '爸爸', '是', '保安'], 27: ['他', '妈妈', '是', '裁缝'], 28: ['他', '外祖父', '是', '邮递', '员'],
             29: ['他', '祖母', '是', '聋人'], 30: ['我', '公公', '是', '门', '卫'], 31: ['你', '妹妹', '是', '会计'],
             32: ['你', '哥哥', '是', '武', '警'], 33: ['你', '婆婆', '是', '盲人'], 34: ['你', '爸爸', '是', '编辑'],
             35: ['她', '外祖父', '是', '农民'], 36: ['我', '儿子', '是', '职员'], 37: ['他', '弟弟', '是', '向导'],
             38: ['他', '岳父', '是', '残疾', '人'], 39: ['他', '姐夫', '是', '刑警'], 40: ['我的', '新', '尺子', '是', '歪', '的'],
             41: ['提高', '他的', '社会', '地位'], 42: ['紧张', '的', '工作', '气氛'], 43: ['工作', '效益', '稳定', '提高'],
             44: ['工作', '环境', '的', '改善'], 45: ['民主', '团结', '的', '局势'], 46: ['我的', '毛毯', '是', '新的'],
             47: ['现实', '情况', '我', '容易', '紧张'], 48: ['形成', '稳定', '的', '基础'], 49: ['社会', '团结', '是', '基础'],
             50: ['我们', '的', '国家', '富强', '民主'], 51: ['国民', '的', '婚姻', '幸福'], 52: ['我们', '是', '自由', '恋爱'],
             53: ['我们', '捐献', '的是', '毛毯'], 54: ['茶壶', '是', '褐色', '的'], 55: ['他的', '工作', '是', '美容'],
             56: ['搞清', '形势', '是', '困难', '的'], 57: ['发挥', '我们', '的', '优势'], 58: ['扭转', '局面', '是', '困难', '的'],
             59: ['现实', '是', '学生', '任务', '多'], 60: ['社会', '的', '安定'], 61: ['地球', '是', '行星'],
             62: ['月亮', '是', '地球', '的', '卫星'], 63: ['太阳', '是', '恒星'], 64: ['国家', '经济', '情况', '好'],
             65: ['他的', '前途', '事业', '成功'], 66: ['他们', '的', '国家', '摆脱', '贫苦'], 67: ['社会', '地位', '的', '提高'],
             68: ['颜色', '是', '丰富', '的'], 69: ['社会', '就业', '形势', '好'], 70: ['他的', '工作', '经验', '丰富'],
             71: ['剪刀', '是', '锋利', '的'], 72: ['新的', '被子', '是', '破', '的'], 73: ['天气', '预报', '有', '雨'],
             74: ['他的', '盆', '是', '绿', '的'], 75: ['杯子', '是', '橙色', '的'], 76: ['加强', '国家', '的', '保卫'],
             77: ['他', '放弃', '目标'], 78: ['引导', '他人', '成功'], 79: ['结果', '圆满', '成功'], 80: ['他的', '手表', '是', '坏', '的'],
             81: ['他的', '牙刷', '疏'], 82: ['洗脸盆', '是', '空', '的'], 83: ['招聘', '的', '岗位', '多'], 84: ['我们', '拜访', '外祖父'],
             85: ['事情', '有', '改善'], 86: ['我们', '的', '友谊', '深'], 87: ['裁缝', '有', '针线'], 88: ['妈妈', '有', '项链'],
             89: ['我有', '打火', '机'], 90: ['天空', '没有', '星星'], 91: ['他的', '小孩子', '有', '礼貌'], 92: ['他', '招呼', '你来'],
             93: ['观察', '他的', '情况'], 94: ['我', '推荐', '他', '去', '就业'], 95: ['手电筒', '有', '电池'],
             96: ['我们', '的', '婚姻', '是', '幸福', '的'], 97: ['国家', '稳定', '是', '幸福', '的', '基础'],
             98: ['社会', '地位', '是', '平等', '的'], 99: ['我的', '毛巾', '是', '干', '的']}


def csv2dict(dataset_type):
    prefix = '../dataset/csl'

    count = 0
    info_dict = dict()
    info_dict['prefix'] = prefix

    print(f"Generate information dict for CSL", dataset_type)

    paths = None
    video_prefix = '/video'
    keypoint_prefix = '/keypoint'

    video_path = prefix + video_prefix
    keypoint_path = prefix + keypoint_prefix

    for idx, sentence in enumerate(sentences):
        # print(file_idx)
        # print(sentences[file_info])
        label = " ".join(["{}"] * len(sentences[sentence])).format(*sentences[sentence])
        class_num = str(idx).zfill(6)

        filenames = os.listdir('{}/{}/'.format(video_path, str(class_num).zfill(6)))
        videos = list(map(lambda x: '{}/{}/{}'.format(video_path, class_num, x), filenames))
        keypoints = list(map(lambda x: '{}/{}/{}'.format(keypoint_path, class_num, x + '.npy'), filenames))

        for file_idx, file_info in tqdm(enumerate(filenames), total=len(filenames)):

            frame_num = np.load(keypoints[file_idx], allow_pickle=True).shape[0]

            file = file_info.split('_')
            signer = file[0]
            info1 = file[1]
            info2 = file[2]
            sample = file[3].replace('.', '')

            if dataset_type == 'all':
                info_dict[count] = {
                    'fileid': file_info,
                    'folder': f"{video_path}/{class_num}",
                    'signer': signer,
                    'label': label,
                    'num_frames': frame_num,
                    'original_info': file_info,
                    'keypoint_path': keypoints[file_idx][1:],
                    'video_path': videos[file_idx][1:],
                    'info1': info1,
                    'info2': info2,
                    'sample': sample
                }
                count = count + 1

            if dataset_type == 'split1-train':
                if sample != '0':
                    info_dict[count] = {
                        'fileid': file_info,
                        'folder': f"{video_path}/{class_num}",
                        'signer': signer,
                        'label': label,
                        'num_frames': frame_num,
                        'original_info': file_info,
                        'keypoint_path': keypoints[file_idx][1:],
                        'video_path': videos[file_idx][1:],
                        'info1': info1,
                        'info2': info2,
                        'sample': sample
                    }
                    count = count + 1

            if dataset_type == 'split1-eval':
                if sample == '0':
                    info_dict[count] = {
                        'fileid': file_info,
                        'folder': f"{video_path}/{class_num}",
                        'signer': signer,
                        'label': label,
                        'num_frames': frame_num,
                        'original_info': file_info,
                        'keypoint_path': keypoints[file_idx][1:],
                        'video_path': videos[file_idx][1:],
                        'info1': info1,
                        'info2': info2,
                        'sample': sample
                    }
                    count = count + 1

            if dataset_type == 'split2-train':
                if signer == 'P50' or \
                        signer == 'P49' or \
                        signer == 'P48' or \
                        signer == 'P47' or \
                        signer == 'P46' or \
                        signer == 'P45' or \
                        signer == 'P44' or \
                        signer == 'P43' or \
                        signer == 'P42' or \
                        signer == 'P41':
                    continue
                else:
                    info_dict[count] = {
                        'fileid': file_info,
                        'folder': f"{video_path}/{class_num}",
                        'signer': signer,
                        'label': label,
                        'num_frames': frame_num,
                        'original_info': file_info,
                        'keypoint_path': keypoints[file_idx][1:],
                        'video_path': videos[file_idx][1:],
                        'info1': info1,
                        'info2': info2,
                        'sample': sample
                    }
                    count = count + 1

            if dataset_type == 'split2-eval':
                if signer == 'P50' or \
                        signer == 'P49' or \
                        signer == 'P48' or \
                        signer == 'P47' or \
                        signer == 'P46' or \
                        signer == 'P45' or \
                        signer == 'P44' or \
                        signer == 'P43' or \
                        signer == 'P42' or \
                        signer == 'P41':
                    info_dict[count] = {
                        'fileid': file_info,
                        'folder': f"{video_path}/{class_num}",
                        'signer': signer,
                        'label': label,
                        'num_frames': frame_num,
                        'original_info': file_info,
                        'keypoint_path': keypoints[file_idx][1:],
                        'video_path': videos[file_idx][1:],
                        'info1': info1,
                        'info2': info2,
                        'sample': sample
                    }
                    count = count + 1

            if dataset_type == 'split2-eval':
                if signer == 'P50' or \
                        signer == 'P49' or \
                        signer == 'P48' or \
                        signer == 'P47' or \
                        signer == 'P46' or \
                        signer == 'P45' or \
                        signer == 'P44' or \
                        signer == 'P43' or \
                        signer == 'P42' or \
                        signer == 'P41':
                    info_dict[count] = {
                        'fileid': file_info,
                        'folder': f"{video_path}/{class_num}",
                        'signer': signer,
                        'label': label,
                        'num_frames': frame_num,
                        'original_info': file_info,
                        'keypoint_path': keypoints[file_idx][1:],
                        'video_path': videos[file_idx][1:],
                        'info1': info1,
                        'info2': info2,
                        'sample': sample
                    }
                    count = count + 1

            if dataset_type == 'test':
                fileid = file_info

                if fileid == 'P41_s9_01_0._color.avi' or fileid == 'P41_s10_02_0._color.avi':
                    info_dict[count] = {
                        'fileid': file_info,
                        'folder': f"{video_path}/{class_num}",
                        'signer': signer,
                        'label': label,
                        'num_frames': frame_num,
                        'original_info': file_info,
                        'keypoint_path': keypoints[file_idx][1:],
                        'video_path': videos[file_idx][1:],
                        'info1': info1,
                        'info2': info2,
                        'sample': sample
                    }
                    count = count + 1


    # print(info_dict)
    print(count)
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


def run_mp_cmd(processes, process_func, process_args):
    with Pool(processes) as p:
        outputs = list(tqdm(p.imap(process_func, process_args), total=len(process_args)))
    return outputs


def run_cmd(func, args):
    return func(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data process for Visual Alignment Constraint for Continuous Sign Language Recognition.')
    parser.add_argument('--dataset', type=str, default='csl',
                        help='save prefix')
    parser.add_argument('--dataset-root', type=str, default='../dataset/csl',
                        help='path to the dataset')

    args = parser.parse_args()
    # mode = ["all", 'split1-train', 'split1-eval', 'split2-train', 'split2-eval']
    mode = ['split1-train', 'split1-eval', 'split2-train', 'split2-eval', 'test']
    sign_dict = dict()
    if not os.path.exists(f"./{args.dataset}"):
        os.makedirs(f"./{args.dataset}")
    for md in mode:
        # generate information dict
        information = csv2dict(dataset_type=md)
        np.save(f"./{args.dataset}/{md}_info.npy", information)

        # update the total gloss dict
        sign_dict_update(sign_dict, information)
        # generate groudtruth stm for evaluation

        generate_gt_stm(information, f"./{args.dataset}/{args.dataset}-groundtruth-{md}.stm")

        # # resize images
        # video_index = np.arange(len(information) - 1)
        # # print(f"Resize image to {args.output_res}")
        # if args.process_image:
        #     if args.multiprocessing:
        #         run_mp_cmd(10, partial(resize_dataset, dsize=args.output_res, info_dict=information), video_index)
        #     else:
        #         for idx in tqdm(video_index):
        #             run_cmd(partial(resize_dataset, dsize=args.output_res, info_dict=information), idx)
    sign_dict = sorted(sign_dict.items(), key=lambda d: d[0])
    save_dict = {}
    dict_count = 0
    for idx, (key, value) in enumerate(sign_dict):
        save_dict[key] = [idx + 1, value]
        dict_count = dict_count + 1
    np.save(f"./{args.dataset}/gloss_dict.npy", save_dict)
    print(dict_count)

# if sample == '0':
#     info_dict_split_1[count_split_1] = {
#         'fileid': file_info,
#         'folder': f"{video_path}/{class_num}",
#         'signer': signer,
#         'label': label,
#         'num_frames': frame_num,
#         'original_info': file_info,
#         'keypoint_path': keypoints[file_idx],
#         'video_path': videos[file_idx],
#         'info1': info1,
#         'info2': info2,
#         'sample': sample
#     }
#     count_split_1 = count_split_1 + 1
#
# if signer == 'P50' or \
#         signer == 'P49' or \
#         signer == 'P48' or \
#         signer == 'P47' or \
#         signer == 'P46' or \
#         signer == 'P45' or \
#         signer == 'P44' or \
#         signer == 'P43' or \
#         signer == 'P42' or \
#         signer == 'P41':
#     info_dict_split_2[count_split_2] = {
#         'fileid': file_info,
#         'folder': f"{video_path}/{class_num}",
#         'signer': signer,
#         'label': label,
#         'num_frames': frame_num,
#         'original_info': file_info,
#         'keypoint_path': keypoints[file_idx],
#         'video_path': videos[file_idx],
#         'info1': info1,
#         'info2': info2,
#         'sample': sample
#     }
#     count_split_2 = count_split_2 + 1