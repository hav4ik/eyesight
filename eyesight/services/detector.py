import cv2
import numpy as np

from ..engine.base_service import BaseService
from ..engine.adapters import get as get_adapter
from ..utils.generic_utils import Resource
from ..utils.output_utils import BBox, DetectionObject, draw_objects
from ..utils import backend_utils as backend

if backend._USING_TENSORFLOW_TFLITE:
    import tensorflow.lite as tflite
elif backend._USING_TFLITE_RUNTIME:
    import tflite_runtime.interpreter as tflite


def input_size(interpreter):
    """Returns input image size as (width, height) tuple."""
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    return width, height


def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, 3)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]


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
    tensor[:h, :w] = np.reshape(resize((w, h)), (h, w, channel))
    return scale, scale


def output_tensor(interpreter, i):
    """Returns output tensor view."""
    tensor = interpreter.tensor(interpreter.get_output_details()[i]['index'])()
    return np.squeeze(tensor)


def get_output(interpreter, score_threshold, image_scale=(1.0, 1.0)):
    """Returns list of detected objects."""
    boxes = output_tensor(interpreter, 0)
    class_ids = output_tensor(interpreter, 1)
    scores = output_tensor(interpreter, 2)
    count = int(output_tensor(interpreter, 3))

    width, height = input_size(interpreter)
    image_scale_x, image_scale_y = image_scale
    sx, sy = width / image_scale_x, height / image_scale_y

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return DetectionObject(
                id=int(class_ids[i]),
                score=float(scores[i]),
                bbox=BBox(xmin=xmin,
                          ymin=ymin,
                          xmax=xmax,
                          ymax=ymax).scale(sx, sy).map(int))

    return [make(i) for i in range(count) if scores[i] >= score_threshold]


def load_labels(path, encoding='utf-8'):
    """Loads labels from file (with or without index numbers).

    Args:
        path: path to label file.
        encoding: label file encoding.
    Returns:
        Dictionary mapping indices to labels.
    """
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}

        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            eyesightrs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in eyesightrs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}


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


class ObjectDetector(BaseService):
    """Object deteciton service, based on MobileNet V2 SSD Coco
    """

    # The neural network should be loaded only once
    interpreter = None
    labels = None

    def __init__(self, camera, *args, **kwargs):
        if not (backend._USING_TFLITE_RUNTIME or
                backend._USING_TENSORFLOW_TFLITE):
            raise NotImplementedError(
                    'TensorflowLite not found. Install it with '
                    '`pip install tflite_runtime`.')

        if not backend._USING_EDGE_TPU:
            raise NotImplementedError(
                    'libedgetpu for Coral Edge TPU not found. Make sure '
                    'that you have it installed.')

        self.interpreter = None
        self.labels = None

        super().__init__(
                adapter=get_adapter('simple')({'cam': camera}),
                *args, **kwargs)

    def load_model(self):
        model_path = Resource(
                collection_name='mobilenet_ssd_v2_coco',
                url='https://github.com/google-coral/'
                    'edgetpu/raw/master/test_data/'
                    'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
        ).path()
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()

    def load_labels(self):
        labels_path = Resource(
                collection_name='mobilenet_ssd_v2_coco',
                url='https://dl.google.com/coral/canned_models/coco_labels.txt'
        ).path()
        self.labels = load_labels(labels_path)

    def _generator(self):
        if self.interpreter is None:
            self.load_model()
        if self.labels is None:
            self.load_labels()

        while True:
            # image preprocess
            image = self._get_inputs('cam')
            scale = set_input(
                    self.interpreter,
                    (image.shape[1], image.shape[0]),
                    lambda size: cv2.resize(image, size))

            # inference
            self.interpreter.invoke()
            objs = get_output(self.interpreter, 0.4, scale)

            # output
            draw_objects(image, objs, self.labels)
            yield image
