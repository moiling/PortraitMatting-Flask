# -*- coding: UTF-8 -*-
import base64

from flask import Flask, request
from gevent import pywsgi
import json
import numpy as np
import warnings
import cv2
from portrait_matting.matting import Matting

warnings.filterwarnings('ignore')

app = Flask(__name__)

M = Matting(checkpoint_path='./checkpoints/ckpt.pt', gpu=False)


@app.route('/api/matting', methods=['POST'])
def matting():
    """
    输入一张图像，返回抠图换底后的图像
    @param image: 图像矩阵数组，类型为array，shape为 w * h * 3
    @param bg: 背景颜色，类型为array，shape为 3，默认值为白色[255, 255, 255]
    @return: result: 抠图换底后的图像矩阵数组，类型为array，shape为 w * h * 3
    """
    data = request.get_json()
    image_numpy = np.array(json.loads(data['image']), dtype=np.uint8)
    try:
        bg_color =  np.array(json.loads(data['bg']), dtype=np.uint8)
    except Exception:
        bg_color = np.array([255, 255, 255])

    matte, img, _ = M.matting(image_numpy, return_img_trimap=True, img_size_in_net=480, img_size_return=-1)
    cut = M.cutout(img, matte)
    comp = M.composite(cut, bg_color / 255.)
    comp = cv2.cvtColor(np.uint8(comp * 255), cv2.COLOR_RGB2BGR)

    res = {'result': json.dumps(comp.tolist())}
    return json.dumps(res)


@app.route('/api/matting_base64', methods=['POST'])
def matting_base64():
    """
    输入一张图像，返回抠图换底后的图像
    @param image: 图像base64字符串
    @param bg: 背景颜色，类型为array，shape为 3，默认值为白色[255, 255, 255]
    @return: result: 图像base64字符串
    """
    data = request.get_json()
    image_base64 = base64.b64decode(data['image'])
    image_numpy = cv2.imdecode(np.fromstring(image_base64, np.uint8), cv2.IMREAD_COLOR)

    try:
        bg_color =  np.array(json.loads(data['bg']), dtype=np.uint8)
    except Exception:
        bg_color = np.array([255, 255, 255])

    matte, img, _ = M.matting(image_numpy, return_img_trimap=True, img_size_in_net=480, img_size_return=-1)
    cut = M.cutout(img, matte)
    comp = M.composite(cut, bg_color / 255.)
    comp = cv2.cvtColor(np.uint8(comp * 255), cv2.COLOR_RGB2BGR)

    comp_base64 = cv2.imencode('.png', comp)[1].tostring()
    comp_base64 = base64.b64encode(comp_base64)

    res = {'result': comp_base64.decode('utf-8')}
    return json.dumps(res)


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 9111), app)
    server.serve_forever()

