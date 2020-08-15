#!/usr/bin/env python
# coding: utf-8
"""
Dual Spatial Pyramid for exposing face warp artifacts in DeepFake videos (DSP-FWA)
"""
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
from py_utils.DL.pytorch_utils.models.classifier import SPPNet
from tqdm import tqdm
import random

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
            rois = roid.cuda()
        rois = rois.permute((0, 3, 1, 2))
        prob = net(rois - bgr_mean)
        prob = F.softmax(prob, dim=1)
        prob = prob.data.cpu().numpy()
        prob = 1 - np.mean(np.sort(prob[:, 0])[np.round(sample_num / 2).astype(int):])
    return prob, face_info


def draw_face_score(im, face_info, prob, threshold):
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
    model_name = 'SPPNet'
    cv2.putText(im, f'Model:{model_name}', (50, 50), font, 1, (0, 255, 255), 3, cv2.LINE_AA)
    return im


async def dsp_fwa_inference(video_path, model_path, output_path, threshold,
                            start_frame=0, end_frame=None, cuda=False):
    print('Starting: {}'.format(video_path))

    video_fn = video_path.split('/')[-1].split('.')[0]+'.mp4'
    os.makedirs(output_path, exist_ok=True)

    sample_num = 10
    front_face_detector = dlib.get_frontal_face_detector()
    lmark_predictor = dlib.shape_predictor('./pretrained_model/shape_predictor_68_face_landmarks.dat')
    print('Loaded front_face_detector & lmark_predictor')

    input_size = 224
    # load network
    net = SPPNet(backbone=50, num_class=2)
    # net = net.cuda()
    net.eval()

    # load training weight
    if cuda:
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['net'])

    # mp4 file path
    imgs, num_frames, fps, width, height = pv.parse_vid(video_path)
    probs = []
    frame = 0
    # reader = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = None
    writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps,
                             (height, width)[::-1])

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame - start_frame)

    try:
        inference_num = 30
        sample_frames = [random.randint(start_frame, end_frame) for _ in range(inference_num)]
        sample_frames.extend([0])
        sample_frames = sorted(set(sample_frames))
        logger.info(f'sample_frames:{sample_frames}')
        logger.info(f'imgs:{type(imgs)}')
    except Exception as e:
        logger.info(e)
    # print('Begin tqdm.....')

    try:
        #for fid in sample_frames:
        for fid, im in enumerate(imgs):
            #frame_num += 1
            #im = imgs[fid]

            if frame_num < start_frame:
                continue
            pbar.update(1)

            if fid in sample_frames:
            # print('Frame: ' + str(fid))
                prob, face_info = im_test(front_face_detector, lmark_predictor, sample_num, net, im, input_size, cuda)
                im = draw_face_score(im, face_info, prob, threshold)
            else:
                im = draw_face_score(im, face_info, prob, threshold)
            # if frame_num >= end_frame:
            #     break

            # # for testing 
            # if frame_num > 10:
            #     break

            writer.write(im)
    except Exception as e:
        print(e)
    pbar.close()
    
    if writer is not None:
        writer.release()
        print('Finished! Output saved under {}'.format(output_path))
    else:
        print('Input video file was empty')

if __name__ == '__main__':
    model_path = './pretrained_model/SPP-res50.pth'
    video_path = './predict/data/clb_fake_id9_id2_0000.mp4'
    output_path = './output/'
    threshold = 0.5
    cuda = False
    start_frame = 0
    end_frame = None

    dsp_fwa_inference(video_path, model_path, output_path, threshold, start_frame, end_frame, cuda)
