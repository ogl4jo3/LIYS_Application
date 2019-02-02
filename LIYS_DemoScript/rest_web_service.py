#!flask/bin/python
from flask import Flask
from flask import request

import cv2
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
from PIL import Image
from io import BytesIO
import base64
import json
import time
import numba as nb

from Segmentation.DeepLabModels import DeepLabModel
from StyleTransfer.StyleModel import StyleModel

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"

# RGBA
colormap = np.array([[0, 0, 0, 0],
                     [128, 0, 0, 128]], dtype=np.uint8)

MODEL_FILE = "Segmentation/models/deeplab_dm0.3_inference.pb"
DEVICE_TYPE = "/gpu:0"
SEG_MODEL = DeepLabModel(MODEL_FILE, DEVICE_TYPE)

styles = ["LIYS_candy", "LIYS_cubist", "LIYS_denoised_starry",
          "LIYS_feathers", "LIYS_mosaic", "LIYS_scream", "LIYS_wave"]

STY_MODEL = StyleModel("StyleTransfer/models/" + styles[0] + ".pb",
                       resize_ratio=1.0,
                       device_t=DEVICE_TYPE)


@app.route('/getStyleInfo')
def getStyleInfo():
    style_images = []

    for style in styles:
        image = Image.open("StyleTransfer/models/" + style + ".jpg")
        buffered = BytesIO()
        image = image.resize((80,80),Image.BILINEAR)
        image.save(buffered, format="JPEG")
        img_bytes = base64.urlsafe_b64encode(buffered.getvalue()).decode('utf-8')
        style_images.append(img_bytes)

    return json.dumps({"styles" : styles, "images" : style_images})

@app.route('/initStyleModel', methods=['POST'])
def initStyleModel():
    print("initStyleModel")

    content = request.get_json()
    style = content['style']
    print(style)

    styModelStart = time.time()
    global STY_MODEL
    STY_MODEL = StyleModel("StyleTransfer/models/" + style + ".pb",
                           resize_ratio=0.5,
                           device_t=DEVICE_TYPE)

    print("model init cost: {:.2f} ms".format((time.time() - styModelStart) * 1000.))

    return str(0)


buffered = BytesIO()

@app.route('/LIYS', methods=['POST'])
def postLIYS():
    print("LIYS")
    content = request.get_json()

    frameTimestamp = time.time()

    image = np.frombuffer(base64.urlsafe_b64decode(content['inputs']), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]

    styStart = time.time()
    global STY_MODEL
    styledImage = STY_MODEL.run(image)
    styledImage = cv2.resize(styledImage, (width, height), interpolation=cv2.INTER_LINEAR)

    styEnd = time.time()
    print("style transfer cost: {:.2f} ms".format((styEnd - styStart) * 1000.))

    segStart = time.time()

    _, seg_map = SEG_MODEL.run(image)

    seg_map = np.array([[0, 0, 0],
                        [1, 1, 1]], dtype=np.uint8)[seg_map]

    seg_map = cv2.resize(seg_map, (width, height), interpolation=cv2.INTER_NEAREST)


    segEnd = time.time()
    print("segmentation cost: {:.2f} ms".format((segEnd - segStart) * 1000.))

    Mergestart = time.time()
    liys_img = image * seg_map + styledImage * (1 - seg_map)
    print("Merge cost: {:.2f} ms".format((time.time() - Mergestart) * 1000.))
    
    liys_img= Image.fromarray(liys_img)

    buffered.truncate(0)
    buffered.seek(0)
    liys_img.save(buffered, format="JPEG")
    img_bytes = base64.urlsafe_b64encode(buffered.getvalue()).decode('utf-8')
    print("Time elapsed {:.2f} ms".format((time.time() - frameTimestamp) * 1000.))
    return json.dumps({"result": img_bytes})


@app.route('/segmentation', methods=['POST'])
def postSeg():
    print("segmentation")
    content = request.get_json()

    frameTimestamp = time.time()
    image = np.frombuffer(base64.urlsafe_b64decode(content['inputs']), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    print(image.shape)
    resized_im, seg_map = SEG_MODEL.run(image)
    seg_image = colormap[seg_map]
    seg_image= Image.fromarray(seg_image, 'RGBA')

    buffered = BytesIO()
    seg_image.save(buffered, format="PNG")
    img_bytes = base64.urlsafe_b64encode(buffered.getvalue()).decode('utf-8')
    print("Time elapsed {:.2f} ms".format((time.time() - frameTimestamp) * 1000.))
    return json.dumps({"result": img_bytes})


@app.route('/styleTransfer', methods=['POST'])
def postSty():
    print("styleTransfer")
    content = request.get_json()

    frameTimestamp = time.time()
    image = np.frombuffer(base64.urlsafe_b64decode(content['inputs']), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    print(image.shape)

    styledImage = STY_MODEL.run(image)
    styledImage= Image.fromarray(styledImage)

    buffered = BytesIO()
    styledImage.save(buffered, format="JPEG")
    img_bytes = base64.urlsafe_b64encode(buffered.getvalue()).decode('utf-8')
    print("Time elapsed {:.2f} ms".format((time.time() - frameTimestamp) * 1000.))
    return json.dumps({"result": img_bytes})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False)




