import cv2
import sys
from time import sleep
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
import os
from moviepy.editor import VideoFileClip
import numba as nb
import sys, getopt
import re

from Segmentation.DeepLabModels import DeepLabModel
from StyleTransfer.StyleModel import StyleModel

""" 
 python transform_LIYS_video.py -i examples/test.mp4 -o examples/LIYS_result.mp4 --sty=StyleTransfer/models/LIYS_mosaic.pb --seg=Segmentation/models/deeplab_dm0.5_multi-scale_inference.pb -w 480 -h 270

 styModelPath ['StyleTransfer/models/LIYS_candy.pb', 'StyleTransfer/models/LIYS_cubist.pb'
            , 'StyleTransfer/models/LIYS_denoised_starry.pb', 'StyleTransfer/models/LIYS_feathers.pb'
            , 'StyleTransfer/models/LIYS_mosaic.pb', 'StyleTransfer/models/LIYS_scream.pb'
            , 'StyleTransfer/models/LIYS_wave.pb']
 segModelPath ["Segmentation/models/deeplab_dm0.3_inference.pb"
        , "Segmentation/models/deeplab_dm0.3_multi-scale_inference.pb"
        , "Segmentation/models/deeplab_dm0.5_inference.pb"
        , "Segmentation/models/deeplab_dm0.5_multi-scale_inference.pb"
        , "Segmentation/models/deeplab_dm1.0_inference.pb"]
"""
argv = sys.argv[1:]
inputVideoPath = ''
outputVideoPath = 'LIYS_result.mp4'
styModelPath = ''
segModelPath = ''
outputWidth = 1280
outputHeight = 720

try:
    opts, args = getopt.getopt(argv,"i:o:sty:seg:w:h:",["sty=","seg="])
except getopt.GetoptError:
    print('python transform_LIYS_video.py -i <inputVideoPath> -o <outputVideoPath> --sty=<styModelPath> --seg=<segModelPath> -w <outputWidth> -h <outputHeight>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-help':
        print('python transform_LIYS_video.py -i <inputVideoPath> -o <outputVideoPath> --sty=<styModelPath> --seg=<segModelPath> -w <outputWidth> -h <outputHeight>')
        sys.exit()
    elif opt in ("-i", "--inputVideoPath"):
        inputVideoPath = arg
    elif opt in ("-o", "--outputVideoPath"):
        outputVideoPath = arg
    elif opt in ("-sty", "--sty"):
        styModelPath = arg
    elif opt in ("-seg", "--seg"):
        segModelPath = arg
    elif opt in ("-w", "--outputWidth"):
        outputWidth = int(arg)
    elif opt in ("-h", "--outputHeight"):
        outputHeight = int(arg)
        
print('inputVideoPath: ', inputVideoPath)
print('outputVideoPath: ', outputVideoPath)
print('styModelPath: ', styModelPath)
print('segModelPath: ', segModelPath)
print('outputWidth: ', outputWidth)
print('outputHeight: ', outputHeight)

DEVICE_TYPE = "/gpu:0"
styModel = StyleModel(styModelPath,
                   resize_ratio=0.5,
                   device_t=DEVICE_TYPE)

# BGR
colormap = np.array([[0, 0, 0],
                     [0, 0, 128]], dtype=np.uint8)

segModel = DeepLabModel(segModelPath, DEVICE_TYPE)

video_capture = cv2.VideoCapture(inputVideoPath)

if (video_capture.isOpened() == False): 
	print("Unable to read video feed")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

print("Input Resolution {0}x{1}".format(frame_width, frame_height))
 
output_video_name = 'LIYS_out.mp4'
# As file at filePath is deleted now, so we should check if file exists or not not before deleting them
if os.path.exists(output_video_name):
    print("check existed file, " + output_video_name)
    sys.exit(2)
    os.remove(output_video_name)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter(output_video_name,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (outputWidth,outputHeight))
frameCnt = 0
totalTime = 0 # unit: second
while(True):
    ret, frame = video_capture.read()

    if ret == True:
        frameStartTime = time.time()
        print("frame count: " + str(frameCnt))
        
        frame = cv2.resize(frame, (outputWidth, outputHeight), interpolation=cv2.INTER_LINEAR)

        styledImage = styModel.run(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        styledImage = cv2.resize(styledImage, (outputWidth, outputHeight),
                             interpolation=cv2.INTER_LINEAR)
        styledImage = cv2.cvtColor(styledImage, cv2.COLOR_RGB2BGR)

        _, seg_map = segModel.run(frame)

        seg_map = np.array([[0, 0, 0],
                        [1, 1, 1]], dtype=np.uint8)[seg_map]
        seg_map = cv2.resize(seg_map, (outputWidth, outputHeight), interpolation=cv2.INTER_NEAREST)
        
        # @nb.jit
        # def trans_seg_img_fun(width, height):
        #     for i in range(height):
        #         for j in range(width):
        #             if (seg_map[i][j] != 0) :
        #                 styledImage[i][j] = frame[i][j]
        #     return styledImage
        # trans_seg_img = trans_seg_img_fun(outputWidth,outputHeight)
        # trans_seg_img = np.array(trans_seg_img)
        # trans_seg_img = trans_seg_img.reshape(outputHeight, outputWidth, 3)

        liys_img = frame * seg_map + styledImage * (1 - seg_map)

        frameCost = time.time() - frameStartTime
        print("one frame cost: {:.2f} ms".format(frameCost * 1000.))

        totalTime += frameCost
 
    	# Write the frame into the file 'output.avi'
        out.write(liys_img)

    	# Display the resulting frame    
		# cv2.imshow('frame',frame)
        frameCnt += 1
 
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
	# Break the loop
    else:
        break
 
# When everything done, release the video capture and video write objects
video_capture.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows() 

avgCost = "{:.2f}".format((totalTime / frameCnt) * 1000.)
print("avg cost: " + avgCost + "ms")

# write audio
clip1 = VideoFileClip(inputVideoPath)
audioclip1 = clip1.audio
clip2 = VideoFileClip(output_video_name)
new_video = clip2.set_audio(audioclip1)
# new_video.write_videofile("LIYS_result_" + avgCost + ".mp4")
if os.path.exists(outputVideoPath):
    print("delete existed file, " + outputVideoPath)
    os.remove(output_video_name)
new_video.write_videofile(outputVideoPath)

os.remove(output_video_name)




