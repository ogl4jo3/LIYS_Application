import numpy as np
import tensorflow as tf
import cv2

""" DeepLabModel """
class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

    def __init__(self, model_path, device_t="/cpu:0"):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        with open(model_path, 'rb') as file_handle:
            graph_def = tf.GraphDef.FromString(file_handle.read())


        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default(), self.graph.device(device_t):
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph,
                               config=tf.ConfigProto(allow_soft_placement=True))

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

            Returns:
                resized_image: RGB image resized from original input image.
                seg_map: Segmentation map of `resized_image`.
        """

        height, width = image.shape[0:2]
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)

        resized_image = cv2.resize(image, (0, 0),
                                   fx=resize_ratio, fy=resize_ratio,
                                   interpolation=cv2.INTER_AREA)

        target_height, target_width = resized_image.shape[0:2]


        batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME,
                                      feed_dict={self.INPUT_TENSOR_NAME: np.expand_dims(resized_image, axis=0)})

        seg_map = batch_seg_map[0]

        return resized_image, seg_map


""" DeepLabModel without preprocessing and postprocessing """
class DeepLabModelTransformed(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'sub_7:0'
    OUTPUT_TENSOR_NAME = 'ArgMax:0'
    INPUT_SIZE = 513

    def __init__(self, model_path, device_t="/cpu:0"):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        self.input_tensor = np.zeros([1, self.INPUT_SIZE, self.INPUT_SIZE, 3], dtype=np.float32)

        graph_def = None
        with open(model_path, 'rb') as file_handle:
            graph_def = tf.GraphDef.FromString(file_handle.read())


        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default(), self.graph.device(device_t):
            tf.import_graph_def(graph_def, name='')

        config = tf.ConfigProto(allow_soft_placement=True)

        self.sess = tf.Session(graph=self.graph, config=config)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

            Returns:
                resized_image: RGB image resized from original input image.
                seg_map: Segmentation map of `resized_image`.
        """

        height, width = image.shape[0:2]
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)

        resized_image = cv2.resize(image, (0, 0),
                                   fx=resize_ratio, fy=resize_ratio,
                                   interpolation=cv2.INTER_AREA)

        target_height, target_width = resized_image.shape[0:2]

        self.input_tensor[:, :, :, :] = 128.0
        self.input_tensor[:, 0:target_height, 0:target_width, :] = resized_image
        self.input_tensor *= 0.00784313771874
        self.input_tensor -= 1.0

        batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME,
                                      feed_dict={self.INPUT_TENSOR_NAME: self.input_tensor})

        seg_map = batch_seg_map[0][0:target_height, 0:target_width]

        return resized_image, seg_map

if __name__ == '__main__':
    print('DeepLabModels')


