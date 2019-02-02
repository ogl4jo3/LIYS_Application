import cv2
import sys
from time import sleep
import numpy as np
import tensorflow as tf
from datetime import datetime
from Segmentation.DeepLabModels import DeepLabModelTransformed


video_capture = cv2.VideoCapture(0)


video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

height, width = map(int, [video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                          video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)])

print("Resolution {0}x{1}".format(height, width))

# BGR
colormap = np.array([[0, 0, 0],
                     [0, 0, 128]], dtype=np.uint8)


MODEL_FILE = "Segmentation/models/frozen_inference_graph_transformed.pb"
DEVICE_TYPE = "/gpu:0"

MODEL = DeepLabModelTransformed(MODEL_FILE, DEVICE_TYPE)


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

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resized_im, seg_map = MODEL.run(frame)

    seg_image = colormap[seg_map]
    resized_im = cv2.cvtColor(resized_im, cv2.COLOR_RGB2BGR)

    overlapping = cv2.addWeighted(resized_im, 1.0, seg_image, 0.7, 0)

    cv2.putText(overlapping,
                "Time elapsed {} ms".format((datetime.now() - frameTimestamp).microseconds / 1000.),
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)

    # Display the resulting frame
    cv2.imshow('Video', overlapping)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
