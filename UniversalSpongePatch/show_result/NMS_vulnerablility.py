import torch
import torchvision
import random
from time import time
import matplotlib.pyplot as plt

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


# 加载一个干净样本的YOLO前向推理后的结果（tensor格式）
output = torch.load('show_result/clean_forward_torch')
output = output.cuda()
# print(output.size())    # (25200, 85)
# print(output)

# 求出每个框的置信度和坐标
output[:, 5:] = output[:, 4:5] * output[:, 5:]
conf, j = output[:, 5:].max(1, keepdim=True)
box_location = xywh2xyxy(output[:, :4])
output = torch.cat([box_location, conf, j.float()], 1)
# print(output.size())    # (25200, 6)


def NMS_forward(output, iou_thres, is_trick=False):
    trick = is_trick    # 是否使用坐标技巧
    c = output[:, 5:6] * (0 if trick else 4096)
    boxes = output[:, :4] + c
    scores = output[:, 4]
    start = time()
    for i in range(10):
        torchvision.ops.nms(boxes, scores, iou_thres)
    return int((time() - start) / 10 * 1000)


# 进入NMS的定位框数量
nums = [1, 10, 100, 1000, 5000, 8000, 10000, 12000, 15000, 18000, 20000, 22000, 25000, 28000, 30000]
times_worst = []
times_best = []
times_rand = []
 
for num in nums:
    # 最坏情况
    ids = random.choices(range(output.size()[0]), k=num)
    output_worst = output[ids, :]
    # 最好情况,所有定位框都重叠
    output_best = output[:1, :].repeat(num, 1)
    # 随机情况
    output_rand = output[ids, :]

    times_worst.append(NMS_forward(output_worst, 1.0))
    times_best.append(NMS_forward(output_best, 0))
    times_rand.append(NMS_forward(output_rand, 0.45))


fig, ax = plt.subplots()

ax.plot(nums, times_worst, 'd', color='r', label='worst')
ax.plot(nums, times_best, 'd', color='g', label='best')
ax.plot(nums, times_rand, 'd', color='b', label='rand')

ax.set_xlabel('Objects[#]')
ax.set_ylabel('NMS time[ms]')
ax.legend()

plt.savefig('./NMS_time.png')
# plt.show()