import numpy as np
import torch
import torchvision.transforms as transforms
import cv2


def repeat_fill(patch, h_real, w_real):

    patch_h, patch_w = patch.shape[1:]
    h_num = h_real // patch_h + 1
    w_num = w_real // patch_w + 1
    patch = patch.repeat(1, h_num, w_num)
    patch = patch[:, :h_real, :w_real]
    return patch

def load_tensor(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape)  # numpy数组格式为（H,W,C）

    tensor = transforms.ToTensor()(img)  # tensor数据格式是torch(C,H,W)
    return tensor


def save_tensor(tensor, path):
    transforms.ToPILImage()(tensor).save(path)


def get_model(name, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'yolov5':
        # taken from https://github.com/ultralytics/yolov5
        from local_yolos.yolov5.models.experimental import attempt_load
        model = attempt_load('local_yolos/yolov5/weights/yolov5s.pt', device).eval()
    elif name == 'yolov3':
        # taken from https://github.com/ultralytics/yolov3
        from local_yolos.yolov3 import hubconf
        model = hubconf.yolov3(pretrained=True, autoshape=False, device=device)
    elif name == 'yolov8':
        # taken from https://github.com/ultralytics/ultralytics
        from local_yolos.yolov5.models.experimental import attempt_load
        model = attempt_load('local_yolos/yolov8s.pt', device).eval()
    return model


def forward(model, images, model_name):
    output = None
    if model_name == 'yolov3' or model_name == 'yolov5':
        output = model(images)[0]
    elif model_name == 'yolov8':
        # 由于YOLOv8的排列顺序不一致，而且每个定位框信息只包含84个（4和坐标信息+80和类别）
        # 没有置信度（类别中的最大值作为置信度）因此为了统一处理，添加一行作为置信度
        output = model(images)[0]
        output = output.transpose(1, 2)
        temp_score = torch.ones_like(output)
        output = torch.cat((output[..., :4], temp_score[..., 0:1], output[..., 4:]), 2)
    return output
