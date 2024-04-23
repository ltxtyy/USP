## 扰动效果图
![picture](https://github.com/ltxtyy/USP/blob/main/UniversalSpongePatch/images/picture.png)

## 训练流程图
![pipeline](https://github.com/ltxtyy/USP/blob/main/UniversalSpongePatch/images/pipeline.png)


## 准备过程
#### 下载数据集（选一个）
* [Berkeley DeepDrive (BDD)](https://doc.bdd100k.com/download.html#k-images)
* [Tsinghua‐Tencent 100K (TT)](https://cg.cs.tsinghua.edu.cn/traffic-sign)
* [PASCAL VOC (VOC 2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)

#### 安装依赖包
`conda create -n USP python=3.9`

`pip install -r requirements.txt`


#### 下载YOLO权重文件（local_yolo/）（选一个）
* [YOLOv3](https://github.com/ultralytics/yolov3/releases/download/v9.6.0/yolov3.pt)
* [YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt)
* [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt)


## 训练过程

`python run_attack.py`


## 致谢
* [YOLOv3](https://github.com/ultralytics/yolov3)，[YOLOv5](https://github.com/ultralytics/yolov5)，[YOLOv8](https://github.com/ultralytics/ultralytics)
* [PhantomSponges](https://github.com/AvishagS422/PhantomSponges)，[UDUP](https://github.com/QRICKDD/UDUP)