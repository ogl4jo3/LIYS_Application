import cv2
import tensorflow as tf

class StyleModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'InputImage:0'
    OUTPUT_TENSOR_NAME = 'StyledImage:0'

    def __init__(self, modelPath, resize_ratio=0.5, device_t="/cpu:0"):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        self.ratio = resize_ratio
        # self.input_tensor = np.zeros([1, int(100 * resize_ratio), int(100 * resize_ratio), 3],
        #                              dtype=np.float32)

        graph_def = None
        with open(modelPath, 'rb') as file_handle:
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
            image: An numpy array object, raw input image.

            Returns:
                styled_image: RGB image styled from original input image.
        """
        height = image.shape[0]
        width = image.shape[1]
        resizeHeight = int(height * self.ratio)
        resizeWidth = int(width * self.ratio)

        input_tensor = cv2.resize(image, (resizeWidth, resizeHeight), interpolation=cv2.INTER_AREA)

        styledImage = self.sess.run(self.OUTPUT_TENSOR_NAME,
                                    feed_dict={self.INPUT_TENSOR_NAME: input_tensor})

        return styledImage


if __name__ == '__main__':
    print('StyleModel')
