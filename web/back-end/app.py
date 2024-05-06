import logging as rel_log
import os
import shutil
from datetime import timedelta, datetime
from flask import *
from processor.AIDetector_pytorch import Detector
import zipfile

import core.main

ALLOWED_EXTENSIONS = {'png', 'jpg', 'zip'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg'}
HOST = '127.0.0.1'
PORT = '5000'
BASE_URL = f'http://{HOST}:{PORT}'
app = Flask(__name__)
app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = r'./uploads'

werkzeug_logger = rel_log.getLogger('werkzeug')
werkzeug_logger.setLevel(rel_log.INFO)

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
app.config['model'] = Detector()


# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    files = request.files.getlist('file')
    assert len(files) == 2

    data_file, model_file = files[0], files[1]
    model = app.config['model']
    # print(data_file.filename, model_file.filename)
    if model_file.filename == 'model.txt':
        # 未传模型，加载默认模型
        model.init_model()
    else:
        model_path = os.path.join('./tmp/model', model_file.filename)
        model_file.save(model_path)
        model.init_model(model_path=model_path)

    if data_file.filename == 'dataset.txt':
        # 未传数据集，使用默认数据集
        filename = 'dataset.zip'
    else:
        filename = data_file.filename

    ext = filename.rsplit('.', 1)[1]
    print(datetime.now(), filename)
    if data_file and ext in ALLOWED_EXTENSIONS:

        image_urls = []
        image_url_base = BASE_URL + '/tmp/img/'
        draw_urls = []
        draw_url_base = BASE_URL + '/tmp/draw/'
        clean_urls = []
        clean_url_base = BASE_URL + '/tmp/clean/'
        image_info = None
        metrics = None

        src_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.isfile(src_path):
            data_file.save(src_path)

        if ext == 'zip':
            # 解压文件夹位置
            directory = filename.rsplit('.', 1)[0]
            # 删除原来的，新建一个空文件夹(防止重名冲突)
            img_dir = os.path.join("tmp/img", directory)
            if os.path.exists(img_dir):
                shutil.rmtree(img_dir)
            os.makedirs(img_dir)
            draw_dir = os.path.join("tmp/draw", directory)
            if os.path.exists(draw_dir):
                shutil.rmtree(draw_dir)
            os.makedirs(draw_dir)
            clean_dir = os.path.join("tmp/clean", directory)
            if os.path.exists(clean_dir):
                shutil.rmtree(clean_dir)
            os.makedirs(clean_dir)

            f = zipfile.ZipFile(src_path, 'r')  # 压缩文件位置
            for file in f.namelist():
                # 只解压png，jpg文件
                if file.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS:
                    f.extract(file, img_dir)  # 解压位置
            f.close()
            image_info, first_file, metrics = core.main.operate_zip(img_dir, model, directory)

            for file in os.listdir(draw_dir):
                if file == first_file:
                    image_urls.insert(0, image_url_base + f'{directory}/' + file)
                    draw_urls.insert(0, draw_url_base + f'{directory}/' + file)
                    clean_urls.insert(0, clean_url_base + f'{directory}/' + file)
                    continue
                image_urls.append(image_url_base + f'{directory}/' + file)
                draw_urls.append(draw_url_base + f'{directory}/' + file)
                clean_urls.append(clean_url_base + f'{directory}/' + file)

        else:
            shutil.copy(src_path, 'tmp/img')
            image_path = os.path.join('tmp/img', filename)
            # print(filename)

            image_info, metrics = core.main.operate_image(image_path, model)

            image_urls.append(image_url_base + filename)
            draw_urls.append(draw_url_base + filename)
            clean_urls.append(clean_url_base + filename)

        return jsonify({'status': 1,
                        'image_urls': image_urls,
                        'draw_urls': draw_urls,
                        'clean_urls': clean_urls,
                        'image_info': image_info,
                        'metrics': metrics})

    return jsonify({'status': 0})



# show photo
@app.route('/tmp/<path:file>', methods=['GET'])
def show_photo(file):
    if request.method == 'GET':
        if not file is None:
            image_data = open(f'tmp/{file}', "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response


@app.route('/')
def hello_world():
    create_dir()
    return redirect(url_for('static', filename='./index.html'))


@app.route("/download", methods=['GET'])
def download_file():
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    return send_from_directory('data', 'testfile.zip', as_attachment=True)


def create_dir():
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    if not os.path.exists('tmp/img'):
        os.mkdir('tmp/img')
    if not os.path.exists('tmp/draw'):
        os.mkdir('tmp/draw')
    if not os.path.exists('tmp/clean'):
        os.mkdir('tmp/clean')
    if not os.path.exists('tmp/model'):
        os.mkdir('tmp/model')



if __name__ == '__main__':
    # with app.app_context():
    #     current_app.model = Detector()
    create_dir()
    app.run(host='127.0.0.1', port=5000, debug=True)
