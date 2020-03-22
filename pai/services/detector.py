import os
import time
import threading

from PIL import Image
from PIL import ImageDraw
import cv2

import tflite_runtime.interpreter as tflite
import platform

import collections
import numpy as np

from pai.services import BaseStreamService
from pai.utils import get_models_path, Log

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

import logging
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.

    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

    @property
    def width(self):
        """Returns bounding box width."""
        return self.xmax - self.xmin

    @property
    def height(self):
        """Returns bounding box height."""
        return self.ymax - self.ymin

    @property
    def area(self):
        """Returns bound box area."""
        return self.width * self.height

    @property
    def valid(self):
        """Returns whether bounding box is valid or not.

        Valid bounding box has xmin <= xmax and ymin <= ymax which is equivalent to
        width >= 0 and height >= 0.
        """
        return self.width >= 0 and self.height >= 0

    def scale(self, sx, sy):
        """Returns scaled bounding box."""
        return BBox(xmin=sx * self.xmin,
                    ymin=sy * self.ymin,
                    xmax=sx * self.xmax,
                    ymax=sy * self.ymax)

    def translate(self, dx, dy):
        """Returns translated bounding box."""
        return BBox(xmin=dx + self.xmin,
                    ymin=dy + self.ymin,
                    xmax=dx + self.xmax,
                    ymax=dy + self.ymax)

    def map(self, f):
        """Returns bounding box modified by applying f for each coordinate."""
        return BBox(xmin=f(self.xmin),
                    ymin=f(self.ymin),
                    xmax=f(self.xmax),
                    ymax=f(self.ymax))

    @staticmethod
    def intersect(a, b):
        """Returns the intersection of two bounding boxes (may be invalid)."""
        return BBox(xmin=max(a.xmin, b.xmin),
                    ymin=max(a.ymin, b.ymin),
                    xmax=min(a.xmax, b.xmax),
                    ymax=min(a.ymax, b.ymax))

    @staticmethod
    def union(a, b):
        """Returns the union of two bounding boxes (always valid)."""
        return BBox(xmin=min(a.xmin, b.xmin),
                    ymin=min(a.ymin, b.ymin),
                    xmax=max(a.xmax, b.xmax),
                    ymax=max(a.ymax, b.ymax))

    @staticmethod
    def iou(a, b):
        """Returns intersection-over-union value."""
        intersection = BBox.intersect(a, b)
        if not intersection.valid:
            return 0.0
        area = intersection.area
        return area / (a.area + b.area - area)


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
        resize: a function that takes a (width, height) tuple, and returns an RGB
            image resized to those dimensions.
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
        return Object(
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
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=[
                tflite.load_delegate(EDGETPU_SHARED_LIB,
                                     {'device': device[0]} if device else {})
            ])


def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                  fill='red')


class ObjectDetector(BaseStreamService):
    camlock = threading.Lock()
    camera = None
    interpreter = None

    def __init__(self, camera):
        """Neural network should be attached to a camera

        TODO: fix this requirement
        """
        ObjectDetector.set_camera(camera)
        super().__init__()

    @classmethod
    def set_camera(cls, camera):
        cls.camlock.acquire()
        cls.camera = camera
        cls.camlock.release()
        Log.info('Camera is set to {}'.format(
            '.'.join([type(camera).__module__, type(camera).__name__])))

    @staticmethod
    def frames():
        root_dir = os.path.join(get_models_path(), 'mobilenet_ssd_v2_coco')
        labels_path = os.path.join(root_dir, 'coco_labels.txt')
        model_path = os.path.join(
                root_dir,
                'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
        if not os.path.isfile(labels_path) or not os.path.isfile(model_path):
            raise RuntimeError('Models not found in {}'.format(root_dir))

        labels = load_labels(labels_path)
        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()

        while True:
            # throw, otherwise we will be locked forever
            if ObjectDetector.camera is None:
                raise RuntimeError(
                        'Neural Network should be attached to a camera')

            # image preprocess
            image = cv2.cvtColor(
                    ObjectDetector.camera.get_frame()[1], cv2.COLOR_BGR2RGB)
            scale = set_input(interpreter, (image.shape[1], image.shape[0]),
                              lambda size: cv2.resize(image, size))

            # inference
            interpreter.invoke()
            objs = get_output(interpreter, 0.4, scale)

            # output
            image = Image.fromarray(image).convert('RGB')
            draw_objects(ImageDraw.Draw(image), objs, labels)
            yield np.array(image)
