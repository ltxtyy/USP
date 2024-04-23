import torch
import torchvision
import cv2
import os
from time import time
import random
import numpy
from torch.utils.data import DataLoader, SubsetRandomSampler

from datasets.augmentations1 import train_transform
from datasets.split_data_set_combined import CustomDataset
from local_yolos.yolov5.utils.general import xywh2xyxy, non_max_suppression
from util.tool import load_tensor, save_tensor, get_model, forward
from attack.sponge_attack import IoU

import matplotlib.pyplot as plt

seed_value = 17
numpy.random.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_dir = 'bdd-dataset/data/train'
# image_dir = 'TT100K/data/train'
# image_dir = 'VOCdevkit/VOC2012/JPEGImages'


def collate_fn(batch):
    return tuple(zip(*batch))

# 获取数据集
idx = random.choices(range(5000), k=1000)
dataset_train = CustomDataset(type_ds='train', img_dir=image_dir, transform=train_transform)
loader = DataLoader(dataset_train, batch_size=8, sampler=SubsetRandomSampler(idx),  collate_fn=collate_fn)

(images, _) = next(iter(loader))
images = torch.stack(images).to(device)

# 加载模型
model_name = 'yolov5'
model = get_model(model_name)


# 获得干净扰动和随机扰动做参照
def get_clean_rand():
    if not os.path.exists('experiments/clean/final_results/'):
        os.makedirs('experiments/clean/final_results/')
    if not os.path.exists('experiments/rand/final_results/'):
        os.makedirs('experiments/rand/final_results/')
    # 干净扰动
    save_tensor(torch.zeros([3, 640, 640]), 'experiments/clean/final_results/final_patch.png')
    # 随机扰动
    save_tensor(torch.rand([3, 640, 640])/5, 'experiments/rand/final_results/final_patch.png')


def metrics(path, images_clean=images):
    conf_thres = 0.25
    iou_thres = 0.45
    # 获取扰动图像(gpu需要warm-up，第一次使用有很大误差不算入结果)
    adv_tensor = torch.zeros([3, 640, 640]) if path == '' else load_tensor(path)
    adv_tensor = adv_tensor.to(device)
    images_adv = torch.clamp(images_clean[:] + adv_tensor, 0, 1)
    images_adv = images_adv.to(device)
    batch_size = images_adv.shape[0]
    # =========forward==========
    count = 0
    total_time = time()
    nms_time = 0
    # output_adv = model(images_adv)[0].detach()
    output_adv = forward(model, images_adv, model_name).detach()

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
        agnostic = False    # NMS trick
        c = x_p[:, 5:6] * (0 if agnostic else 4096)  # classes
        boxes, scores = x_p[:, :4] + c, x_p[:, 4]  # boxes (offset by class), scores
        
        count += len(x_p)

        nms_time_start = time()
        nms_i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        nms_time += time() - nms_time_start

    total_time = time() - total_time
    # print(f"total time: {round(total_time, 3)*1000}ms, NMS time: {round(nms_time, 3)*1000}ms, pass to NMS: {count // batch_size}", end='\t')
    

    images_clean = images_clean.to(device)
    output_clean = forward(model, images_clean, model_name).detach()

    pred_clean_bboxes = non_max_suppression(output_clean, conf_thres, iou_thres, max_det=10000)
    pred_adv_bboxes = non_max_suppression(output_adv, conf_thres, iou_thres, max_det=10000)

    iou = IoU(conf_thres, iou_thres, (640, 640), device)

    original_count = 0
    reserve_count = 0

    for (img_clean_preds, img_adv_preds) in zip(pred_clean_bboxes, pred_adv_bboxes):
        original_count += img_clean_preds.shape[0]

        box_xyxy_clean = (img_clean_preds[:, :4]).to(device)
        box_xyxy_adv = (img_adv_preds[:, :4]).to(device)
        res = iou.get_iou(box_xyxy_clean, box_xyxy_adv)
        res_max = res.max(dim=1)[0]

        # IoU > 0.45证明是同一个框
        c = ((res_max > 0.45) + 0).sum().item()
        reserve_count += c

    recall = round(reserve_count/original_count, 3)
    # print(f"Recall: {recall*100}%")
    return total_time, nms_time, count // batch_size, recall


# 多次实验取平均值
def multiple_experiments(path, count_time):
    total_time, nms_time, count, recall = 0, 0, 0, 0
    for i in range(count_time):
        _total_time, _nms_time, _count, _recall = metrics(path)
        total_time += _total_time
        nms_time += _nms_time
        count += _count
        recall += _recall
    total_time, nms_time, count, recall = total_time / count_time, nms_time / count_time, count / count_time, recall / count_time
    return total_time, nms_time, count, recall



# 获得每种配置的最终扰动对应的指标
def get_final():
    res = []
    path = 'experiments/'
    for conf in os.listdir(path):
        dir = os.path.join(path, conf)
        if not os.path.isdir(dir):
            continue
        n = 5
        config = [
                'clean',
                f'yolov_{n}_epsilon=30_lambda1=0.8_lambda2=10', 
                f'yolov_{n}_epsilon=30_lambda1=1_lambda2=10', 
                f'yolov_{n}_epsilon=70_lambda1=0.8_lambda2=10', 
                f'yolov_{n}_epsilon=70_lambda1=1_lambda2=10',

                # 'yolov_5_epsilon=70_lambda1=1_lambda2=10',
                # 'yolov_3_5_epsilon=70_lambda1=1_lambda2=10',
                # 'yolov_3_5_8_epsilon=70_lambda1=1_lambda2=10',
              ]
        if not (conf in config):
            continue
        image_path = os.path.join(dir, 'final_results/final_patch.png')

        # 实验十次取平均值
        total_time, nms_time, count, recall = multiple_experiments(image_path, 10)

        s = f'{round(total_time, 3)*1000}ms({round(nms_time, 3)*1000}ms)/{count}/{round(recall, 3)*100}%'
        print(conf + '\t' + s)

    # 按顺序打印
    #     if conf == 'clean' or conf == 'rand':
    #         print(conf + '\t' + s)
    #     else:
    #         res.append([k.split('=')[1] for k in conf.split('_')[-3:]] + [s])

    # res.sort(key=lambda x : (x[0], x[1], x[2]))
    # for l in res:
    #     print(l)

if __name__ == '__main__':
    # 生成干净样本和随机样本做对比
    get_clean_rand()

    # GPU需要warm-up，第一次使用可能有很大误差不算入结果
    metrics('')
    get_final()