import torch
import os
import cv2
from time import time

from util.tool import load_tensor, get_model, forward
from local_yolos.yolov5.utils.general import xywh2xyxy, non_max_suppression
from datasets.augmentations1 import train_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'yolov5'
model = get_model(model_name, device)
adv = load_tensor('experiments/yolov_5_epsilon=70_lambda1=1_lambda2=10/final_results/final_patch.png')
adv = adv.to(device)

video_dir = 'video/1'
# print(os.listdir(video_dir))
# 排序帧图像，按顺序输入模型
ls = sorted(os.listdir(video_dir), key=lambda x: int(x.split('.')[0]))
start = time()
for path in ls:
    img_path = os.path.join(video_dir, path)
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image = train_transform(image=image)['image'].float()
    image = image.unsqueeze(0)
    image = image.to(device)
    # image = torch.clamp(image[:] + adv, 0, 1)

    output = forward(model, image, model_name).detach()
    non_max_suppression(output)
    # post_process(output)
    # break
period = time() - start
FPS = len(ls) / period
print('Time: ', period)
print('FPS: ', FPS)