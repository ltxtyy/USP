import cv2
import os
import numpy as np


def predict(data_path, file_name, model, is_save=True):
    data_path = data_path.replace('\\', '/')
    x = cv2.imread(data_path)
    img_y, img_clean, image_info, res = model.detect(x)
    if is_save:
        cv2.imwrite('./tmp/draw/{}'.format(file_name), img_y)
        cv2.imwrite('./tmp/clean/{}'.format(file_name), img_clean)
    return image_info, res


def operate_image(path, model):
    file_name = os.path.split(path)[1]
    image_info, res_metrics = predict(path, file_name, model)

    return image_info, res_metrics


def average(name, metrics):
    # 计算所有图片对于每个patch的平均值，再加上配置名称
    avg = list(np.mean(metrics, axis=0).round(1))
    res = [[name[i]] + list(avg[i]) for i in range(len(avg))]
    return res


def operate_zip(img_dir, model, directory, max_len=10):
    res_metrics = []
    res_name = None
    first_info = None
    first_file = None
    count = 0
    for file in os.listdir(img_dir):
        path = os.path.join(img_dir, file)
        if os.path.isfile(path):
            image_info, metrics = predict(path, os.path.join(directory, file),
                                          model, count < max_len)
            count += 1
            res_metrics.append([s[1:] for s in metrics])
            if first_info is None:  # 只拿第一个信息展示
                first_info = image_info
                first_file = file
                res_name = [s[0] for s in metrics]
    return first_info, first_file, average(res_name, res_metrics)


if __name__ == '__main__':
    pass
