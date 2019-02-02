import cv2
import sys
from time import sleep
import numpy as np
import tensorflow as tf
from datetime import datetime
from StyleTransfer.StyleModel import StyleModel

video_capture = cv2.VideoCapture(0)


video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

height, width = map(int, [video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                          video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)])

print("Resolution {0}x{1}".format(height, width))

style = {"LIYS_candy" : "StyleTransfer/models/LIYS_candy.pb",
         "LIYS_cubist" : "StyleTransfer/models/LIYS_cubist.pb",
         "LIYS_denoised_starry" : "StyleTransfer/models/LIYS_denoised_starry.pb",
         "LIYS_feathers" : "StyleTransfer/models/LIYS_feathers.pb",
         "LIYS_mosaic" : "StyleTransfer/models/LIYS_mosaic.pb",
         "LIYS_scream" : "StyleTransfer/models/LIYS_scream.pb",
         "LIYS_wave" : "StyleTransfer/models/LIYS_wave.pb"}

MODEL = StyleModel(style["LIYS_candy"],
                   resize_ratio=0.5,
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

    styledImage = MODEL.run(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    styledImage = cv2.cvtColor(styledImage, cv2.COLOR_RGB2BGR)

    styledImage = cv2.resize(styledImage, (width, height),
                             interpolation=cv2.INTER_LINEAR)

    cv2.putText(styledImage,
                "Time elapsed {} ms".format((datetime.now() - frameTimestamp).microseconds / 1000.),
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)

    # Display the resulting frame
    cv2.imshow('Video', styledImage)
    # cv2.imshow('Orig Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()