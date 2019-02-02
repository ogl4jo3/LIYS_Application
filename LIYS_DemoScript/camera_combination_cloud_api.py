from __future__ import print_function
import cv2
import sys
from time import sleep
import numpy as np
import tensorflow as tf
from datetime import datetime

from PIL import Image
from grpc.beta import implementations
import matplotlib.pyplot as plt
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import requests
from io import StringIO, BytesIO
import time

import base64
import json
import argparse

video_capture = cv2.VideoCapture(0)


video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

height, width = map(int, [video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                          video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)])

#print("Resolution {0}x{1}".format(height, width))


# BGR
colormap = np.array([[0, 0, 0],
                     [0, 0, 128]], dtype=np.uint8)

load_seg_model_start = time.time()

server = '34.80.179.216:8500'
host, port = server.split(':')



while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frameTimestamp = datetime.now()

    test_time_end = time.time()
    start = time.time()
    style_start = time.time()
    
    image = Image.fromarray(frame)
    width, height = image.size
    resize_ratio = 1.0 * 513 / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

    
    buffered = BytesIO()
    resized_image.save(buffered, format="JPEG")
    # urlsafe_
    # img_bytes = base64.b64encode(img_array.tobytes()).decode('utf-8')
    img_bytes = base64.urlsafe_b64encode(buffered.getvalue()).decode('utf-8')
    # img_bytes = base64.b64encode(open('8U5FpQ.jpg', "rb").read()).decode('utf-8')
    #print(len(img_bytes))
    
    instance = [{"b64": img_bytes}]
    data = json.dumps({"inputs": img_bytes})
    #print(data)
    
    #print("Image shape:", resized_image.size)
    
    json_response = requests.post("http://34.80.179.216:8501/v1/models/deeplab:predict", data=data)
    
    # Extract text from JSON
    response = json.loads(json_response.text)
    #print(response)
    # Interpret bitstring output
    response = json.loads(json_response.text)

    #print(type(response['outputs']))
    #print(response['outputs']['b64'][:20])


    mask = np.frombuffer(base64.urlsafe_b64decode(response['outputs']['b64']), np.uint8)
    #print(mask.shape)
    mask = cv2.imdecode(mask, 0)
    #print(mask.shape)

    
    
    
    #/-----------------------------------------------------------------------/
    # fill in the request object with the necessary data

    image_mask = mask.reshape(target_size[1], target_size[0])
    
    seg_image = colormap[image_mask]
    nparray_resized_image = np.asarray(resized_image)
    #print(nparray_resized_image.shape)
    #print(seg_image.shape)
    resized_im = cv2.cvtColor(nparray_resized_image, cv2.COLOR_RGB2BGR)
    overlapping = cv2.addWeighted(nparray_resized_image, 1.0, seg_image, 0.7, 0)

    
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    #resized_im = cv2.resize(resized_im, (width, height), interpolation=cv2.INTER_LINEAR)
    #resized_im = cv2.cvtColor(resized_im, cv2.COLOR_RGB2BGR)
    
    end = time.time()
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,20)
    fontScale              = 0.6
    fontColor              = (0,0,255)
    lineType               = 2
    cv2.putText(overlapping,
                "Time elapsed {:.2f} ms".format((end - start) * 1000.),
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
    
    # Display the resulting frame
    cv2.imshow('Video', overlapping)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
    #print("------ load seg model = {:.2f} ------".format(load_seg_model_end - load_seg_model_start))
    #print("------ load style model = {:.2f} ------".format(load_style_model_end - load_style_model_start))
    #print("------ style_time = {:.2f} ------".format(style_end-style_start))
    #print("------ seg_time = {:.2f} ------".format(seg_end-seg_start))
    #print("------ time = {:.2f} ------".format(end-start))
    #print("------ test time = {:.2f}".format(test_time_end - test_time_start))
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

