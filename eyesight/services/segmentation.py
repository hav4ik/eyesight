import cv2
import numpy as np
import imutils
import time
# https://scikit-image.org/docs/stable/auto_examples/edges/plot_polygon.html
from skimage.measure import approximate_polygon

from ..engine.base_service import BaseService
from ..engine.adapters import get as get_adapter
from ..utils.generic_utils import Resource, log
from ..utils import backend_utils as backend

if backend._USING_TENSORFLOW_TFLITE:
    import tensorflow.lite as tflite
elif backend._USING_TFLITE_RUNTIME:
    import tflite_runtime.interpreter as tflite


def set_input(interpreter, size, resize):
    """Copies a resized and properly zero-padded image to the input tensor.

    Args:
        interpreter: Interpreter object.
        size: original image size as (width, height) tuple.
        resize: a function that takes a (width, height) tuple, and returns an
            RGB image resized to those dimensions.
    Returns:
        Actual resize ratio, which should be passed to `get_output` function.
    """
    width, height = input_size(interpreter)
    w, h = size
    scale = min(width / w, height / h)
    w, h = int(w * scale), int(h * scale)
    tensor = input_tensor(interpreter)
    tensor.fill(0)    # padding
    _, _, channel = tensor.shape
    img = resize((w, h))

    # TODO: find a CPU model that doesn't require input normalization
    if not backend._USING_EDGE_TPU:
        img = (img.astype(np.float32) - 127.5) / 127.5

    tensor[:h, :w] = np.reshape(img, (h, w, channel))
    return (scale, scale), img


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    try:
        if backend._USING_EDGE_TPU:
            interpreter = tflite.Interpreter(
                    model_path=model_file,
                    experimental_delegates=[
                        tflite.load_delegate(
                            backend._EDGETPU_SHARED_LIB,
                            {'device': device[0]} if device else {})
                    ])
        else:
            interpreter = tflite.Interpreter(model_path=model_file)

    except ValueError:
        interpreter = tflite.Interpreter(model_path=model_file)
    return interpreter


def input_size(interpreter):
    """Returns input image size as (width, height) tuple."""
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    return width, height


def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, 3)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]


def output_tensor(interpreter, i):
    """Returns output tensor view."""
    tensor = interpreter.tensor(interpreter.get_output_details()[i]['index'])()
    return np.squeeze(tensor)


class SemanticSegmentator(BaseService):
    def __init__(self, camera, *args, **kwargs):
        self.interpreter = None

        super().__init__(
                adapter=get_adapter('simple')({'cam': camera}),
                *args, **kwargs)

    def load_model(self):
        if backend._USING_EDGE_TPU:
            model_path = Resource(
                    collection_name='mobilenet_v2_deeplab_v3_pascal2012',
                    url='https://github.com/google-coral/'
                        'edgetpu/raw/master/test_data/'
                        'deeplabv3_mnv2_pascal_quant_edgetpu.tflite'
            ).path()
        else:
            model_path = Resource(
                    collection_name='mobilenet_v2_deeplab_v3_pascal2012',
                    url='https://storage.googleapis.com/'
                        'download.tensorflow.org/models/tflite/gpu/'
                        'deeplabv3_257_mv_gpu.tflite'
            ).path()

        # Create an interpreter and allocate memory for it
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()

    def _generator(self):
        if self.interpreter is None:
            self.load_model()

        while True:
            image = self._get_inputs('cam')
            scale, img = set_input(
                    self.interpreter,
                    (image.shape[1], image.shape[0]),
                    lambda size: cv2.resize(image, size))

            self.interpreter.invoke()
            result = output_tensor(self.interpreter, 0)
            result = result.copy()[
                    :int(image.shape[0] * scale[0]),
                    :int(image.shape[1] * scale[1])]

            # TODO: find a model that doesn't need this softmax
            if not backend._USING_EDGE_TPU:
                result = np.argmax(result, axis=2)

            yield result


class SegmentationExtrapolator(BaseService):
    def __init__(self, segmentator, tracker, orig_size=None):
        self.tmp_canvas = None
        self.prev_cntrs = None
        self.orig_size = orig_size
        self.cntrs = None

        input_services = {
            'trck': tracker,
            'segm': segmentator,
        }
        super().__init__(adapter=get_adapter('latest')(input_services))

    def _process_class(self, sx, sy):
        raw_cntrs = cv2.findContours(
                self.tmp_canvas, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        raw_cntrs = imutils.grab_contours(raw_cntrs)

        self.cntrs = []
        for cntr in raw_cntrs:
            approx_cntr = cv2.approxPolyDP(
                    cntr.reshape(-1, 2), epsilon=2.5, closed=True)
            # approx_cntr = cntr
            approx_cntr[..., 0] = approx_cntr[..., 0] * sx
            approx_cntr[..., 1] = approx_cntr[..., 1] * sy
            self.cntrs.append(approx_cntr)

    def _generator(self):
        while True:
            segmentation = self._get_inputs('segm')
            assert isinstance(segmentation, np.ndarray) and \
                len(segmentation.shape) == 2

            segmentation = cv2.resize(
                    segmentation, (0, 0), fx=0.5, fy=0.5,
                    interpolation=cv2.INTER_NEAREST)

            if self.tmp_canvas is None or \
                    self.tmp_canvas.shape != segmentation.shape:
                self.tmp_canvas = np.zeros(
                        segmentation.shape, dtype=np.uint8)

            begin = time.time()
            self.tmp_canvas[...] = 0
            self.tmp_canvas[segmentation == 15] = 255
            sx, sy = 1, 1
            if self.orig_size is not None:
                sx = self.orig_size[0] / segmentation.shape[1]
                sy = self.orig_size[1] / segmentation.shape[0]

            self._process_class(sx, sy)
            end = time.time()
            # log.warn('perm={}, fps={}'.format(
            #     end - begin, 1. / (end - begin)))
            yield self.cntrs
