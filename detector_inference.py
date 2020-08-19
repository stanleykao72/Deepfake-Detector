#!/usr/bin/env python
# coding: utf-8
import time
import torch
import torch.nn.functional as F
import cv2, os, dlib
from os.path import join
import logging
import numpy as np
import pandas as pd
from py_utils.face_utils import lib
from py_utils.vid_utils import proc_vid as pv
from py_utils.DL.sppnet.models.classifier import SPPNet
from py_utils.DL.efficientnet.models.classifiers import DeepFakeClassifier  ## for EfficientNet
from py_utils.DL.xceptionnet.models import model_selection   ## for Xception Network

from tqdm.auto import tqdm
import random
import re

from skimage import io
from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from interpretability.guided_back_propagation import GuidedBackPropagation
from interpretability.utils import (get_last_conv_name, prepare_input,
                                    gen_cam, norm_image, gen_gb, save_image)

import logging

LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
DATE_FORMAT = '%Y%m%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger(__name__)


def im_test(front_face_detector, lmark_predictor, sample_num, net, im, input_size, cuda):

    # print('begin im_test.....')
    face_info = lib.align(im[:, :, (2,1,0)], front_face_detector, lmark_predictor)
    # print('face_info finished.....')
    # Samples
    if len(face_info) != 1:
        prob = -1
    else:
        _, point = face_info[0]
        rois = []
        for i in range(sample_num):
            roi, _ = lib.cut_head([im], point, i)
            rois.append(cv2.resize(roi[0], (input_size, input_size)))

        # vis_ = np.concatenate(rois, 1)
        # cv2.imwrite('vis.jpg', vis_)

        bgr_mean = np.array([103.939, 116.779, 123.68])
        bgr_mean = bgr_mean[np.newaxis, :, np.newaxis, np.newaxis]
        bgr_mean = torch.from_numpy(bgr_mean).float()
        if cuda:
            bgr_mean = bgr_mean.cuda()
        
        rois = torch.from_numpy(np.array(rois)).float()
        if cuda:
            rois = rois.cuda()
        rois = rois.permute((0, 3, 1, 2))
        prob = net(rois - bgr_mean)
        prob = F.softmax(prob, dim=1)
        prob = prob.data.cpu().numpy()
        prob = 1 - np.mean(np.sort(prob[:, 0])[np.round(sample_num / 2).astype(int):])
    return prob, face_info


def draw_face_score(model_name, im, face_info, prob, threshold):
    if len(face_info) == 0:
        return im

    _, points = np.array(face_info[0], dtype=object)
    x1 = np.min(points[:, 0])
    x2 = np.max(points[:, 0])
    y1 = np.min(points[:, 1])
    y2 = np.max(points[:, 1])

    # Fake: (0, 255, 0), Real: (0, 0, 255)
    if prob >= threshold:
        label = 'fake'
        color = (0, 0, 255)
    else:
        label = 'real'
        color = (0, 255, 0)
    #label = 'fake' if prob >= 0.5 else 'real'
    #color = (0, (1 - prob) * 255, prob * 255)
    cv2.rectangle(im, (x1, y1), (x2, y2), color, 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, f'{prob:.3f}=>{label}', (x1, y1 - 10), font, 1, color, 3, cv2.LINE_AA)
    cv2.putText(im, f'Model:{model_name}', (50, 50), font, 1, (0, 255, 255), 3, cv2.LINE_AA)
    return im


def load_network_xception(model_path, cuda=True):
    # Load network
    net = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    # load training weight
    if cuda:
        net.load_state_dict(torch.load(model_path))
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))

    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    if cuda:
        net = net.cuda()
    # logger.info(f'loaded Xception model {net}')
    return net


def load_network_sppnet(model_path, cuda=True):
    try:
        # load network
        net = SPPNet(backbone=50, num_class=2)
        if cuda:
            net = net.cuda()
        net.eval()
    except Exception as e:
        logger.info(f'load network:{e}')
    try:
        # load training weight
        if cuda:
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        logger.info(checkpoint.keys())
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
    except Exception as e:
        logger.info(f"load weight:{e}")
    # logger.info(f'loaded SPPNet model {net}')
    return net


def load_network_efficientnet(model_path, cuda=True):
    try:
        # load network
        net = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns")
        if cuda:
            net = net.cuda()
        net.eval()
    except Exception as e:
        logger.info(f'load network:{e}')
    try:
        # load training weight
        if cuda:
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        logger.info(checkpoint.keys())
        state_dict = checkpoint.get("state_dict", checkpoint)
        net.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
    except Exception as e:
        logger.info(f"load weight:{e}")
    # logger.info(f'loaded SPPNet model {net}')
    return net


def grad_cam(frame, model_name, cam_model, net, im, im_size, class_idx=None, cuda=False):

    # img = np.float32(cv2.resize(im, (im_size, im_size))) / 255
    img = np.float32(im)/255
    inputs = prepare_input(img)
    if cuda:
        inputs = inputs.cuda()

    # print(inputs.shape)
    # 输出图像
    image_dict = {}

    layer_name = get_last_conv_name(net)
    # print(layer_name)

    # Grad-CAM
    if cam_model == 'GradCAM':
        grad_cam = GradCAM(net, layer_name)
        mask = grad_cam(inputs, class_idx)  # cam mask
        # mask = cv2.resize(mask, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
        image_dict['GradCAM'], image_dict['heatmap'] = gen_cam(img, mask)
        grad_cam.remove_handlers()
    # Grad-CAM++
    if cam_model == 'GradCAMpp':
        #logger.info('1')
        grad_cam_plus_plus = GradCamPlusPlus(net, layer_name)
        #logger.info('2')

        mask_plus_plus = grad_cam_plus_plus(inputs, class_idx)  # cam mask
        # mask_plus_plus = cv2.resize(mask_plus_plus, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
        #logger.info('3')

        image_dict['GradCAMpp'], image_dict['heatmap++'] = gen_cam(img, mask_plus_plus)
        #logger.info('4')

        grad_cam_plus_plus.remove_handlers()
        #logger.info('5')


#     # GuidedBackPropagation
#     gbp = GuidedBackPropagation(net)
#     inputs.grad.zero_()  # 梯度置零
#     grad = gbp(inputs)
    
#     gb = gen_gb(grad)
#     image_dict['gb'] = norm_image(gb)
    
#     # 生成Guided Grad-CAM
#     cam_gb = gb * mask[..., np.newaxis]
#     image_dict['cam_gb'] = norm_image(cam_gb)
    # save_image(image_dict, str(frame), 'net', './output')

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_dict[cam_model], f'Model:{model_name}', (50, 50), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
    return image_dict[cam_model]


def sample_frames(start_frame, end_frame, num):
    try:
        # num = 30
        sample_list = [random.randint(start_frame, end_frame) for _ in range(num)]
        sample_list.extend([0])
        sample_list = sorted(set(sample_list))
        logger.info(f'sample_frames:{sample_list}')
    except Exception as e:
        logger.info(f'sample_frames:{e}')
    return sample_list


async def detector_inference(model_name, video_path, model_path, output_path, threshold, cam,
                             start_frame=0, end_frame=None, cuda=False):
    logger.info('Starting: {}'.format(video_path))

    cam_model = 'GradCAMpp'
    logger.info('set cam model')

    video_fn = video_path.split('/')[-1].split('.')[0]
    video_fn_cam = f'{video_fn}_{model_name}_{cam_model}.mp4'
    video_fn = f'{video_fn}_{model_name}.mp4'

    os.makedirs(output_path, exist_ok=True)

    sample_num = 10
    front_face_detector = dlib.get_frontal_face_detector()
    lmark_predictor = dlib.shape_predictor('./pretrained_model/shape_predictor_68_face_landmarks.dat')
    logger.info('Loaded front_face_detector & lmark_predictor')

    input_size = 224
    class_idx = 0

    if model_name == 'SPPNet':
        net = load_network_sppnet(model_path, cuda)
        logger.info('Loaded SPPNet')
    if model_name == 'XceptionNet':
        net = load_network_xception(model_path, cuda)
        logger.info('Loaded XceptionNet')
    if model_name == 'EfficientnetB7':
        net = load_network_efficientnet(model_path, cuda)
        logger.info('Loaded EfficientnetB7')

    # mp4 file path
    imgs, num_frames, fps, width, height = pv.parse_vid(video_path)
    probs = []
    frame = 0
    # reader = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = None
    writer_cam = None
    writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps,
                             (height, width)[::-1])
    writer_cam = cv2.VideoWriter(join(output_path, video_fn_cam), fourcc, fps,
                                (height, width)[::-1])


    logger.info(f'num_frames:{num_frames}, fps:{fps}, width:{width}, height:{height}')
    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames

    try:
        if cam:
            cam_list = sample_frames(start_frame, end_frame, 1)
            pbar = tqdm(total=len(cam_list))
            for fid, im in enumerate(imgs):
                if fid in cam_list:
                    pbar.update(1)
                    #if fid <= 30:
                    cam_im = grad_cam(fid, model_name, cam_model, net, im, input_size, class_idx, cuda)            
                    writer_cam.write(cam_im)
                    #else:
                    #    break
            pbar.close()

        sample_list = sample_frames(start_frame, end_frame, 30)
        pbar = tqdm(total=end_frame - start_frame)
        for fid, im in enumerate(imgs):
            pbar.update(1)

            if fid in sample_list:

            # print('Frame: ' + str(fid))
                prob, face_info = im_test(front_face_detector, lmark_predictor, sample_num, net, im, input_size, cuda)
                bnd_im = draw_face_score(model_name, im, face_info, prob, threshold)
            else:
                bnd_im = draw_face_score(model_name, im, face_info, prob, threshold)
            writer.write(bnd_im)
            
    except Exception as e:
        logger.info(f'generate image:{e}')
    pbar.close()
    if writer is not None:
        writer.release()
        logger.info('Finished! Output saved under {}'.format(output_path))
    else:
        logger.info('Input video file was empty')
    if writer_cam is not None:
        writer_cam.release()
        logger.info('Finished! Output saved under {}'.format(output_path))
    else:
        logger.info('Input video file was empty')


if __name__ == '__main__':
    # SPPNet
    model_name = 'SPPNet'
    model_path = './pretrained_model/SPP-res50.pth'
    #video_path = './predict/data/data_dst.mp4'
    video_path = './predict/data/Trump_AndyLiu_2.mp4'
    output_path = './output/'
    threshold = 0.5
    cam = False
    cuda = False
    start_frame = 0
    end_frame = None

    detector_inference(model_name, video_path, model_path, output_path, threshold, cam, start_frame, end_frame, cuda)

    # XceptionNet
    model_name = 'XceptionNet'
    model_path = './pretrained_model/df_c0_best.pkl'
    #video_path = './predict/data/data_dst.mp4'
    video_path = './predict/data/Trump_AndyLiu_2.mp4'
    output_path = './output/'
    threshold = 0.5
    cam = True
    cuda = True
    start_frame = 0
    end_frame = None

    detector_inference(model_name, video_path, model_path, output_path, threshold, cam, start_frame, end_frame, cuda)

    # EfficientnetB7
    model_name = 'EfficientnetB7'
    model_path = './pretrained_model/b7_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_0'
    #video_path = './predict/data/data_dst.mp4'
    video_path = './predict/data/Trump_AndyLiu_2.mp4'
    output_path = './output/'
    threshold = 0.5
    cam = True
    cuda = True
    start_frame = 0
    end_frame = None

    detector_inference(model_name, video_path, model_path, output_path, threshold, cam, start_frame, end_frame, cuda)
