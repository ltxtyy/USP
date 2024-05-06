import os

import torch
import torchvision
import numpy as np
from local_yolos.yolov5.utils.general import non_max_suppression, xywh2xyxy, box_iou
from utils.image_tool import image_to_tensor, repeat_fill
import cv2
from time import time
from random import randint
import albumentations as A
from albumentations.pytorch import ToTensorV2
from local_yolos.yolov5.models.experimental import attempt_load


class Detector(object):

    def __init__(self):
        self.img_size = 640
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.max_frame = 160
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = A.Compose(
            [
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255),
                ToTensorV2(),
            ],
        )
        # self.init_model()

    def init_model(self, model_path=None):

        model = None
        if model_path:
            model = attempt_load(model_path, self.device).eval()
        else:
            model = attempt_load('weights/yolov5s.pt', self.device).eval()

        self.model = model
        self.names = model.module.names if hasattr(model, 'module') else model.names
        self.colors = [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in self.names]

    def preprocess(self, img):
        img = self.transform(image=img)['image'].float()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

    def plot_bboxes(self, image, bboxes, line_thickness=None):
        # print('shape:', image.shape)
        h, w, _ = image.shape
        scale_w = w / 640
        scale_h = h / 640
        tl = line_thickness or round(
            0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
        for (x1, y1, x2, y2, _, cls_id) in bboxes:
            x1, y1, x2, y2, cls_id = int(x1 * scale_w), int(y1 * scale_h), \
                                     int(x2 * scale_w), int(y2 * scale_h), int(cls_id)
            color = self.colors[cls_id]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)  # font thickness
            cv2.putText(image, f'{self.names[cls_id]}', (x1, y1 - 2), 0, tl / 3, color,
                        thickness=tf, lineType=cv2.LINE_AA)
        return image

    def metrics(self, adv_tensor, images_clean):
        conf_thres = 0.25
        iou_thres = 0.45
        images_adv = torch.clamp(images_clean[:] + adv_tensor, 0, 1)
        # =========forward==========
        before_count = 0
        nms_time = 0
        after_count = 0
        total_time = time()

        output_adv = self.model(images_adv)[0].detach()

        flag_p = output_adv[..., 4] > conf_thres  # candidates

        for (i, patch) in enumerate(output_adv):
            x_p = patch[flag_p[i]]

            # Compute conf
            x_p[:, 5:] = x_p[:, 5:] * x_p[:, 4:5]
            box_p = xywh2xyxy(x_p[:, :4])
            # best class only
            conf, j = x_p[:, 5:].max(1, keepdim=True)
            x_p = torch.cat((box_p, conf, j.float()), 1)[conf.view(-1) > conf_thres]

            # Batched NMS
            agnostic = False  # NMS trick
            c = x_p[:, 5:6] * (0 if agnostic else 4096)  # classes
            boxes, scores = x_p[:, :4] + c, x_p[:, 4]  # boxes (offset by class), scores

            before_count += len(x_p)

            nms_time_start = time()
            after_count += len(torchvision.ops.nms(boxes, scores, iou_thres))  # NMS
            nms_time += time() - nms_time_start

        total_time = time() - total_time

        total_time = round(total_time, 3) * 1000
        nms_time = round(nms_time, 3) * 1000
        # print(f"total time: {total_time}ms, NMS time: {nms_time}ms,"
        #       f" pass to NMS: {before_count}, after NMS: {after_count}", end='\t')

        output_clean = self.model(images_clean)[0].detach()

        pred_clean_bboxes = non_max_suppression(output_clean, conf_thres, iou_thres, max_det=10000)
        pred_adv_bboxes = non_max_suppression(output_adv, conf_thres, iou_thres, max_det=10000)

        original_count = 0
        reserve_count = 0

        for (img_clean_preds, img_adv_preds) in zip(pred_clean_bboxes, pred_adv_bboxes):
            original_count += img_clean_preds.shape[0]

            box_xyxy_clean = img_clean_preds[:, :4]
            box_xyxy_adv = img_adv_preds[:, :4]
            res = box_iou(box_xyxy_clean, box_xyxy_adv)
            res_max = res.max(dim=1)[0]

            # IoU > 0.45证明是同一个框
            c = ((res_max > 0.45) + 0).sum().item()
            reserve_count += c

        recall = round(int((reserve_count / original_count) * 1000) / 10, 1)
        # print(f"Recall: {recall * 100}%")
        return total_time, nms_time, before_count, after_count, recall

    def detect(self, im):

        img = self.preprocess(im)
        h, w = img.shape[2:]

        # 加扰动
        best_adv = ''
        best_before_count = 0
        res = []
        root = 'perturbation'
        for file in os.listdir(root):
            path = os.path.join(root, file)
            patch = image_to_tensor(path)
            adv = repeat_fill(patch, h, w)
            metrics = self.metrics(adv, img)
            if metrics[2] > best_before_count:
                best_before_count = metrics[2]
                best_adv = adv
            res.append([file.split('.')[0], *metrics])
        res.sort(key=lambda x: x[0])

        # 选择攻击效果最好的提取info展示
        image_adv = torch.clamp(img[:] + best_adv, 0, 1)
        pred = self.model(image_adv)[0].detach()
        pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, max_det=20000)[0]
        image_info = []
        count = dict()
        for *coord, conf, cls_id in pred:
            lbl = self.names[int(cls_id)]
            x1, y1 = int(coord[0]), int(coord[1])
            x2, y2 = int(coord[2]), int(coord[3])
            count[lbl] = count.get(lbl, 0) + 1
            image_info.append(['{}-{}'.format(lbl, count[lbl]),
                               '{}×{}'.format(x2 - x1, y2 - y1),
                               np.round(float(conf), 3)])
        image_info.sort(key=lambda x: (x[0].split('-')[0], int(x[0].split('-')[1])))

        im_clean = im.copy()
        im = self.plot_bboxes(im, pred)
        # 画一张干净样本效果图
        output = self.model(img)[0].detach()
        output = non_max_suppression(output, self.conf_threshold, self.iou_threshold, max_det=100)[0]
        im_clean = self.plot_bboxes(im_clean, output)
        return im, im_clean, image_info, res
