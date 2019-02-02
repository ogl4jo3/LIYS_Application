# Live In Your Style
- - -
## Android App
Real-Time Video Segmentation and Stylization on Android Devices
- - -

### Two Solution:
- Native: use tflite or pb model to inference image.
- Cloud: post image to cloud then return inferenced image.

### Getting Started
1. Download Android Studio
2. Launch Android Studio
3. Select "File/Open"
4. Click LIYS_APP folder
5. Click *Run/Run 'app'*.

## Demo Script
- - -
### 1. camera : get camera frame then inference it.
- `camera_person_detect.py`, use deeplab model to segment person and background (pb format model).
- `camera_person_detect_tflite.py`, use deeplab model to segment person and background (TensorFlow Lite format model).
- `camera_style_transfer.py`, use StyleTranfer model to stylize camera frame.
- `camera_combination.py`, use deeplab and StyleTranfer model to segment person and stylize background. 
- `camera_combination_cloud_api.py`, post camera frame to cloud then get inferenced frame (gRPC). 

### 2. transform video : transform video with segmentation and stylization
- `transform_LIYS_video.py`, Run `python transform_LIYS_video.py` to view all the possible parameters. 
- **Flags**
    - `-i`:Path to input video.
    - `-o`:Path to transformed video.
    - `--sty`:style transfer model path.
    - `--seg`:segmentation model path.
    - `-w`:output video width. default 1280.
    - `-h`:output video height. default 720.
- **Example usage**: 
```   
python transform_LIYS_video.py 
-i examples/VIDEO0049_part.mp4 \
-o examples/LIYS_result.mp4 \
--sty=StyleTransfer/models/LIYS_mosaic.pb \ 
--seg=Segmentation/models/deeplab_dm0.5_multi-scale_inference.pb \
-w 480 -h 270
```

### 3. web service : restful web service ([Flask]), Real-Time segmentation and stylization.
- `rest_web_service.py`, Run `python rest_web_service.py`, connect to http://0.0.0.0:5000/ , call the url you need.
- REST APIs:

``` 
- "/getStyleInfo" - [GET]:
    get style name and style thumbnail(base64 encode).
response:{
        styles : ["style1", "style2", ...]
        images : ["style1_thumbnail_base64encode", "style2_thumbnail_base64encode", ...]
}
``` 

``` 
- "/initStyleModel" - [POST]:
    initial style transfer model, before LIYS or styleTransfer.
request:{
        style : "style name"
    }
response: 0
``` 

``` 
- "/LIYS" - [POST]:
    Segment Person and Style Transfer Background.
request:{
        inputs : "frame base64encoded"
}
response:{
        result : "inferenced frame base64encoded"
}
``` 

``` 
- "/segmentation" - [POST]:
    Segment Person.
request:{
        inputs : "frame base64encoded"
}
response:{
        result : "segmented frame base64encoded"
}
``` 

``` 
- "/styleTransfer" - [POST]:
    Style Transfer image.
request:{
        inputs : "frame base64encoded"
}
response:{
        result : "style transfered frame base64encoded"
}
``` 

## Transform Graph and Convert to TFLite (deeplab model)
TensorFlow Lite does not support the preprocessing done within the DeepLabv3 model. Preprocessing must be done offline before the images are run.
- - -
### Transform Graph
remove prepocessing and postprocessing.
```
transform_graph \
    --in_graph=frozen_inference_graph.pb \
    --out_graph=frozen_inference_graph_transformed.pb \
    --inputs="sub_7" \
    --outputs="ArgMax" \
    --transforms='fold_batch_norms fold_old_batch_norms strip_unused_nodes(type=float, shape="1,513,513,3")'
```
### Transform Graph 2
GPU-accelerated computing.(flatten_atrous_conv)
```
transform_graph \
    --in_graph=frozen_inference_graph_transformed.pb \
    --out_graph=frozen_inference_graph_transformed_2.pb \
    --inputs="sub_7" \
    --outputs="ArgMax" \
    --transforms='fold_constants flatten_atrous_conv remove_device merge_duplicate_nodes fold_batch_norms fold_old_batch_norms strip_unused_nodes(type=float, shape="1,513,513,3")'
```
### Convert to TFLite
from pb format model convert to tflite formate.
```
tflite_convert \
    --graph_def_file=frozen_inference_graph_transformed_2.pb \
    --output_file=deeplab.tflite \
    --inference_type=FLOAT \
    --inference_input_type=FLOAT \
    --input_shape=1,513,513,3 \
    --input_array=sub_7 \
    --output_array=ArgMax \
    --output_format=TFLITE
```

## TODOs
- - -
- **App** : 
    - iOS version
    - transform StyleTransfer model to tensorflow lite format.
- **Web** : 
    - HLS instead of restful web service.
    - Amazon Kinesis Video Streams

## References
- - -
1. **Segmentation model**: [DeepLab: Deep Labelling for Semantic Image Segmentation]
2. **Style Transfer model**: [Fast Style Transfer in TensorFlow]
3. **Android TensorFlow(pb) Example**: [TensorFlow Android Camera Demo]
4. **Android TensorFlow Lite Example**: [TF Lite Android Image Classifier App Example]
5. **A suite of tools for modifying computational graphs**: [Graph Transform Tool]
6. **Convert to TensorFlow Lite format**: [TensorFlow Lite]

[Flask]:<http://flask.pocoo.org/>
[DeepLab: Deep Labelling for Semantic Image Segmentation]: <https://github.com/tensorflow/models/tree/master/research/deeplab>
[Fast Style Transfer in TensorFlow]: <https://github.com/lengstrom/fast-style-transfer>
[TensorFlow Android Camera Demo]: <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android>
[TF Lite Android Image Classifier App Example]: <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/java/demo>
[Graph Transform Tool]:<https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms>
[TensorFlow Lite]:<https://www.tensorflow.org/lite/overview>

