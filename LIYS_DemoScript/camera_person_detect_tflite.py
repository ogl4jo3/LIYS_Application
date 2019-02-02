
from time import sleep
import numpy as np
import tensorflow as tf
import cv2
from datetime import datetime



video_capture = cv2.VideoCapture(0)


video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

height, width = map(int, [video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                          video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)])

print("Resolution {0}x{1}".format(height, width))



font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,20)
fontScale              = 0.6
fontColor              = (0,0,255)
lineType               = 2

# BGR
colormap = np.array([[0, 0, 0],
                     [0, 0, 128]], dtype=np.uint8)



# Load TFLite model and allocate tensors.
# MODEL_FILE = "models/deeplab.tflite"
MODEL_FILE = "Segmentation/models/deeplab_dm30.tflite"
interpreter = tf.contrib.lite.Interpreter(model_path=MODEL_FILE)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


input_tensor = np.zeros([1, 513, 513, 3], dtype=np.float32)

resize_ratio = 1.0 * 513 / max(width, height)


while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass


    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frameTimestamp = datetime.now()

    resized_image = cv2.resize(frame, (0, 0),
                               fx=resize_ratio, fy=resize_ratio,
                               interpolation=cv2.INTER_AREA)

    target_height, target_width = resized_image.shape[0:2]

    input_tensor[:, :, :, :] = 128.0
    input_tensor[0, 0:target_height, 0:target_width, :] = \
        cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    input_tensor *= 0.00784313771874
    input_tensor -= 1.0

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    seg_map = interpreter.get_tensor(output_details[0]['index'])[0, :, :, 0]
    seg_map = seg_map[0:target_height, 0:target_width]

    seg_image = colormap[seg_map]

    overlapping = cv2.addWeighted(resized_image, 1.0, seg_image, 0.7, 0)

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
