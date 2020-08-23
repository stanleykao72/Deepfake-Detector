#!/usr/bin/env python
# coding: utf-8
import time
import torch
import torch.nn.functional as F
import cv2
import os
# import dlib
from facenet_pytorch import MTCNN
from os.path import join
import logging
import numpy as np
import pandas as pd
# from py_utils.face_utils import lib
from py_utils.vid_utils import proc_vid as pv
from py_utils.DL.sppnet.models.classifier import SPPNet
from py_utils.DL.efficientnet.models.classifiers import build_model
from py_utils.DL.efficientnet.models.cv_util import isotropically_resize_image
from py_utils.DL.efficientnet.models.cv_util import padding_image
from py_utils.DL.xceptionnet.models import model_selection   ## for Xception Network
from skimage import io

from tqdm.auto import tqdm
import random
import re

# CAM
from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from interpretability.guided_back_propagation import GuidedBackPropagation
from interpretability.utils import (get_last_conv_name, prepare_input,
                                    gen_cam, norm_image, gen_gb, save_image)

import logging

LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
DATE_FORMAT = '%Y%m%d %H:%M:%S'
logging.basicConfig(level=logging.ERROR, format=LOGGING_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger(__name__)


def face_mtcnn(fid, model_name, net, im, input_size, cuda):
    device =  torch.device('cuda:0' if cuda else 'cpu')

    face_detector = MTCNN(margin=0,thresholds=[0.85, 0.95, 0.95], device=device)    
    
    im_h, im_w, _ = im.shape
    logger.info(f'im.shape:h-->{im_h}, w-->{im_w}')
    try:
        try:
            face_info, _ = face_detector.detect(im, landmarks=False)
            #face_info, _, _ = mtcnn.detect(im, landmarks=True)
            logger.info(f'face_info:{face_info}')
        except Exception as e:
            logger.error(f"mtcnn error: {e}")
        
        if face_info is not None:
            xmin, ymin, xmax, ymax = [int(b) for b in face_info[0]]
            logger.info(f'xmin:{xmin}, ymin:{ymin}, xmax:{xmax}, ymax:{ymax}')

            # facenet detector
            w = xmax - xmin
            h = ymax - ymin
            p_h = h // 3
            p_w = w // 3
            y1 = max(ymin - p_h, 0)
            y2 = ymax + p_h
            x1 = max(xmin - p_w, 0)
            x2 = xmax + p_w
            #logger.info(f'(y1:{y1}:y2:{y2},x1:{x1}:x2:{x2})')
            inputs = im[y1:y2, x1:x2]
            inputs_shape = inputs.shape
            #logger.info(f'inputs_shape:{inputs_shape}')

            #face = torch.from_numpy(np.array(face)).float()

    #         # from Owen
            INPUT_SIZE = 380
    #         inputs = isotropically_resize_image(inputs, INPUT_SIZE)
    #         inputs = padding_image(inputs, INPUT_SIZE)
    #         cv2.imwrite(f'./output/inputs_{fid}.jpg', inputs)
    #         inputs = inputs / 255.

            face_h, face_w, _ = inputs.shape
            #logger.info(f'face shape:{inputs.shape}')

            # prepare for cam
            face = inputs.copy()

            # inference
            inputs = inputs[np.newaxis, :, :, :]
            #logger.info(f'inputs:{inputs.shape}')
            inputs = torch.from_numpy(inputs).float()
            inputs = inputs.permute((0, 3, 1, 2))   ## 將tensor的維度進行轉換, 此列為置換維度順序
            if cuda:
                inputs = inputs.cuda()
            with torch.no_grad():
                outputs = net(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy()
            #logger.info(f'outputs:{outputs}')
            del outputs
            prob = probs[0][0]
            #logger.info(f'prob:{prob}')



            # grad_cam
            mask_2 = np.zeros(im.shape[:2], np.uint8)
            ## 製作白色遮罩
            mask_2[y1:y2, x1:x2] = 255
            ## 黑白相反
            mask_inv = cv2.bitwise_not(mask_2)
            masked_img_2 = cv2.bitwise_and(im, im, mask=mask_inv)        
            #logger.info(f'mask:{masked_img_2.shape}')


            # im_size = 224

            for class_idx in [0]:
                cam = grad_cam(fid, 'GradCAMpp', net, face, INPUT_SIZE, class_idx, cuda)
                cam_h, cam_w, _ = cam.shape

            ## 平移
            tx, ty = x1, y1
            M1 = np.float32([[1, 0, tx],   # 向右 tx
                             [0, 1, ty]])  # 向下 ty
            shift_img1 = cv2.warpAffine(cam, M1, (im_w, im_h))  #

            cam_merge = masked_img_2 + shift_img1
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cam_merge, f'Model:{model_name}', (50, 50), font, 1, (0, 255, 255), 3, cv2.LINE_AA)
#             cv2.imwrite(f'./output/cam_merge:{fid}.jpg', cam_merge)

            ## face
            face = [x1, y1, x2, y2]
            #logger.info(f'face_mtcnn:{face}')
        else:
            prob = None
            face = None
    except Exception as e:
        logger.error(f"face_mtcnn error: {e}")

    if cuda:
        torch.cuda.empty_cache()
    return prob, face, cam_merge if face_info is not None else im


def draw_face_score(model_name, im, face_info, prob, threshold):
    font = cv2.FONT_HERSHEY_SIMPLEX
    if face_info is not None:
        x1, y1, x2, y2 = [int(b) for b in face_info]
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
        cv2.putText(im, f'{prob:.3f}=>{label}', (x1, y2 + 50), font, 1, color, 3, cv2.LINE_AA)
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
        # net = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns")
        net, epoch, bce_best = build_model(encoder="tf_efficientnet_b7_ns", weights=model_path, no_spp=False)
        net.module.encoder.eval()
        for p in net.module.encoder.parameters():
            p.requires_grad = False
        if cuda:
            net = net.cuda()
        net.eval()
    except Exception as e:
        logger.info(f'load network:{e}')
    return net


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


def grad_cam(fid, cam_model, net, im, im_size, class_idx=None, cuda=False):

    im_shape = im.shape
    #logger.info(f'grad_cam im_shape:{im_shape}')
    #img = np.float32(cv2.resize(im, (im_size, im_size))) / 255
    img = np.float32(im)/255
    inputs = prepare_input(img)
    if cuda:
        inputs = inputs.cuda()
    #logger.info(f'grad_cam inputs:{type(inputs)}')

    # 输出图像
    image_dict = {}
    
    layer_name = get_last_conv_name(net)
    #logger.info(f'grad_cam layer_name:{layer_name}')


    # Grad-CAM
    if cam_model == 'GradCAM':
        grad_cam = GradCAM(net, layer_name)
        mask = grad_cam(inputs, class_idx)  # cam mask
        # mask = cv2.resize(mask, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
        image_dict['GradCAM'], image_dict['heatmap'] = gen_cam(img, mask)
        grad_cam.remove_handlers()
    # Grad-CAM++
    if cam_model == 'GradCAMpp':
        grad_cam_plus_plus = GradCamPlusPlus(net, layer_name)
        mask_plus_plus = grad_cam_plus_plus(inputs, class_idx)  # cam mask
        image_dict['GradCAMpp'], image_dict['heatmap++'] = gen_cam(img, mask_plus_plus)
#         cv2.imwrite(f'./output/GradCAMpp:{fid}.jpg', image_dict['GradCAMpp'])
#         cv2.imwrite(f'./output/heatmappp:{fid}.jpg', image_dict['heatmap++'])
        grad_cam_plus_plus.remove_handlers()
    return image_dict[cam_model]


async def detector_inference(model_name, video_path, model_path, output_path, threshold, cam, cam_model,
                             predit_video, cam_video, start_frame=0, end_frame=None, cuda=False):
    logger.info('Starting: {}'.format(video_path))

    # cam_model = 'GradCAMpp'
    # logger.info('set cam model')
    video_fileid = video_path.split('/')[-1].split('.')[0]
    if predit_video:
        video_fn = predit_video
    else:
        video_fn = f'{output_path}{video_fileid}_{model_name}.mp4'
        logger.info(f'video_fn:{video_fn}')
    if cam_video:
        video_fn_cam = cam_video
    else:
        video_fn_cam = f'{output_path}{video_fileid}_{model_name}_{cam_model}.mp4'
        logger.info(f'video_fn_cam:{video_fn_cam}')

    os.makedirs(output_path, exist_ok=True)

    input_size = 224
    class_idx = 0

    if model_name == 'SPPNet':
        net = load_network_sppnet(model_path, cuda)
    if model_name == 'XceptionNet':
        net = load_network_xception(model_path, cuda)
    if model_name == 'EfficientnetB7':
        net = load_network_efficientnet(model_path, cuda)

    # mp4 file path
    imgs, num_frames, fps, width, height = pv.parse_vid(video_path)
    probs = []
    frame = 0
    logger.info(f'num_frames:{num_frames}, fps:{fps}, width:{width}, height:{height}')

    # reader = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = None
    writer_cam = None
    writer = cv2.VideoWriter(video_fn, fourcc, fps, (height, width)[::-1])
    writer_cam = cv2.VideoWriter(video_fn_cam, fourcc, fps, (height, width)[::-1])

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames

    try:
        sample_list = sample_frames(start_frame, end_frame, 30)
        pbar = tqdm(total=end_frame - start_frame)
        for fid, im in enumerate(imgs):
            pbar.update(1)

            if fid in sample_list:
                prob, face_info, cam_im = face_mtcnn(fid, model_name, net, im, input_size, cuda)
                bnd_im = draw_face_score(model_name, im, face_info, prob, threshold)
            else:
                bnd_im = draw_face_score(model_name, im, face_info, prob, threshold)
            writer.write(bnd_im)
            writer_cam.write(cam_im)

    except Exception as e:
        logger.error(f'generate image:{e}')
    pbar.close()
    if writer is not None:
        writer.release()
        logger.info(f'Finished! Output saved under {output_path}{video_fn}')
    else:
        logger.info('Input video file was empty')
    if writer_cam is not None:
        writer_cam.release()
        logger.info(f'Finished! Grad-cam Output saved under {output_path}{video_fn_cam}')
    else:
        logger.info('Input video file was empty')


if __name__ == '__main__':
    output_path = './output/'
    threshold = 0.5
    cam = True
    cam_model = 'GradCAMpp'
    cuda = False
    start_frame = 0
    end_frame = None
    video_path = './input/result_Andy_AIA.mp4'

    model_name = 'XceptionNet'
    model_path = './pretrained_model/df_c0_best.pkl'
    detector_inference(model_name, video_path, model_path, output_path, threshold, cam, cam_model, start_frame, end_frame, cuda)

    model_name = 'SPPNet'
    model_path = './pretrained_model/SPP-res50.pth'
    detector_inference(model_name, video_path, model_path, output_path, threshold, cam, cam_model, start_frame, end_frame, cuda)

    model_name = 'EfficientnetB7'
    model_path = './pretrained_model/tf_efficientnet_b7_ns_spp_last'
    detector_inference(model_name, video_path, model_path, output_path, threshold, cam, cam_model, start_frame, end_frame, cuda)
