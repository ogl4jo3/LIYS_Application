import cv2
import sys
from time import sleep
import numpy as np
import tensorflow as tf
from datetime import datetime
import numba as nb
import time
from Segmentation.DeepLabModels import DeepLabModelTransformed
from StyleTransfer.StyleModel import StyleModel

video_capture = cv2.VideoCapture(0)


video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

height, width = map(int, [video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                          video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)])

print("Resolution {0}x{1}".format(height, width))

# BGR
colormap = np.array([[0, 0, 0],
                     [0, 0, 128]], dtype=np.uint8)

style = {"LIYS_candy" : "StyleTransfer/models/LIYS_candy.pb",
         "LIYS_cubist" : "StyleTransfer/models/LIYS_cubist.pb",
         "LIYS_denoised_starry" : "StyleTransfer/models/LIYS_denoised_starry.pb",
         "LIYS_feathers" : "StyleTransfer/models/LIYS_feathers.pb",
         "LIYS_mosaic" : "StyleTransfer/models/LIYS_mosaic.pb",
         "LIYS_scream" : "StyleTransfer/models/LIYS_scream.pb",
         "LIYS_wave" : "StyleTransfer/models/LIYS_wave.pb"}


Seg_MODEL_FILE = "Segmentation/models/frozen_inference_graph_transformed.pb"
DEVICE_TYPE = "/gpu:0"
Seg_MODEL = DeepLabModelTransformed(Seg_MODEL_FILE, DEVICE_TYPE)

style_name = "LIYS_candy"
if len(sys.argv) > 1:
    style_name = sys.argv[1]

Trans_MODEL = StyleModel(style[style_name],
                   resize_ratio=0.2,
                   device_t='/gpu:0')

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,20)
fontScale              = 0.6
fontColor              = (0,0,255)
lineType               = 2


while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frameTimestamp = datetime.now()

    start = time.time()
    style_start = time.time()

    styledImage = Trans_MODEL.run(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    style_end = time.time()
    
    styledImage = cv2.cvtColor(styledImage, cv2.COLOR_RGB2BGR)

    styledImage = cv2.resize(styledImage, (width, height),
                             interpolation=cv2.INTER_LINEAR)


    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    seg_start = time.time()
    resized_im, seg_map = Seg_MODEL.run(frame)
    seg_end = time.time()
    
    resized_im = cv2.resize(resized_im, (width, height), interpolation=cv2.INTER_LINEAR)
    resized_im = cv2.cvtColor(resized_im, cv2.COLOR_RGB2BGR)
    seg_map = cv2.resize(seg_map, (width, height), interpolation=cv2.INTER_NEAREST)

    @nb.jit
    def trans_seg_img_fun(width, height):
        for i in range(height):
            for j in range(width):
                if (seg_map[i][j] != 0) :
                    styledImage[i][j] = resized_im[i][j]
        return styledImage
    
#trans_seg_img = [ resized_im[i][j] if seg_map[i][j]!=0 else styledImage[i][j]  for i in range(height) for j in range(width)]                 
    trans_seg_img = trans_seg_img_fun(width,height)
    trans_seg_img = np.array(trans_seg_img)
    trans_seg_img = trans_seg_img.reshape(height, width, 3)
    seg_image = colormap[seg_map]
    
    end = time.time()
    
    cv2.putText(trans_seg_img,
                "Time elapsed {:.2f} ms".format((end - start) * 1000.),
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)


    # Display the resulting frame
    cv2.imshow('Video', trans_seg_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print("------ style_time = {:.2f} ------".format(style_end-style_start))
    print("------ seg_time = {:.2f} ------".format(seg_end-seg_start))
    print("------ time = {:.2f} ------".format(end-start))

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

