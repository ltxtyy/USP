# 1. 网站效果：

最终效果：

![在这里插入图片描述](https://github.com/ltxtyy/USP/blob/main/web/image/effect-1.png)

上传对应的模型和数据集后的效果：

![在这里插入图片描述](https://github.com/ltxtyy/USP/blob/main/web/image/effect-2.png)

# 2. 启动项目：

在 Flask 后端项目下启动后端代码(再启用前最好新建一个weight文件夹，然后在里面下载yolov5s的模型，即可作为默认模型)：

```bash
python app.py
```

在 VUE 前端项目下，先安装依赖：

```bash
npm install
```

然后运行前端：

```bash
npm run serve
```

然后在浏览器打开即可