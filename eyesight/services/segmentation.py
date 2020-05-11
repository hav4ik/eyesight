import cv2
import numpy as np

from ..engine.base_service import BaseService
from ..engine.adapters import get as get_adapter
from ..utils.generic_utils import Resource
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
    tensor[:h, :w] = np.reshape(img, (h, w, channel))
    return (scale, scale), img


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    try:
        interpreter = tflite.Interpreter(
                model_path=model_file,
                experimental_delegates=[
                    tflite.load_delegate(
                        backend._EDGETPU_SHARED_LIB,
                        {'device': device[0]} if device else {})
                ])
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
        model_path = Resource(
                collection_name='mobilenet_v2_deeplab_v3_pascal2012',
                url='https://github.com/google-coral/'
                    'edgetpu/raw/master/test_data/'
                    'deeplabv3_mnv2_pascal_quant_edgetpu.tflite'
        ).path()
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
            yield result
