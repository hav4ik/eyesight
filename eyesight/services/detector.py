import collections
import platform

import cv2
import tflite_runtime.interpreter as tflite
import numpy as np

from ..engine.base_service import BaseService
from ..utils import Resource

# library loading
EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]

# Structure to hold objects
Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.

    Represents a rectangle which sides are either vertical or horizontal,
    parallel to the x or y axis.
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

        Valid bounding box has xmin <= xmax and ymin <= ymax which is
        equivalent to width >= 0 and height >= 0.
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
                        EDGETPU_SHARED_LIB,
                        {'device': device[0]} if device else {})
                ])
    except ValueError:
        interpreter = tflite.Interpreter(model_path=model_file)
    return interpreter


def text_over(img,
              text,
              bottomleft,
              tx_color=(0, 0, 0),
              bg_color=(0, 255, 0),
              thickness=1,
              padding=(5, 5),
              font_scale=0.6,
              font=cv2.FONT_HERSHEY_DUPLEX):

    (tw, th), baseline = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=thickness)

    ymax = min(max(bottomleft[1], th + 2 * padding[1]), img.shape[0])
    xmin = max(min(bottomleft[0], img.shape[1] - tw - 2 * padding[0]), 0)
    box_coords = (
            (xmin, ymax),
            (xmin + tw + 2 * padding[0], ymax - th - 2 * padding[1]))
    cv2.rectangle(img, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(img, text, (xmin + padding[0], ymax - padding[1]),
                font, fontScale=font_scale,
                color=tx_color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_objects(img, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        cv2.rectangle(img, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax),
                      (0, 255, 0), 2)
        text_over(img,
                  '{:s} ({:.1f}%)'.format(
                      labels.get(obj.id, obj.id), obj.score * 100),
                  (bbox.xmin, bbox.ymin))


class ObjectDetector(BaseService):
    """Object deteciton service, based on MobileNet V2 SSD Coco
    """

    # The neural network should be loaded only once
    interpreter = None
    labels = None

    def __init__(self,
                 camera,
                 inactivity_timeout=10,
                 client_timeout=5):
        super().__init__(
                input_services={'camera': camera},
                inactivity_timeout=inactivity_timeout,
                client_timeout=inactivity_timeout)

    @staticmethod
    def load_model():
        model_path = Resource(
                collection_name='mobilenet_ssd_v2_coco',
                url='https://github.com/google-coral/'
                    'edgetpu/raw/master/test_data/'
                    'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
        ).path()
        ObjectDetector.interpreter = make_interpreter(model_path)
        ObjectDetector.interpreter.allocate_tensors()

    @staticmethod
    def load_labels():
        labels_path = Resource(
                collection_name='mobilenet_ssd_v2_coco',
                url='https://dl.google.com/coral/canned_models/coco_labels.txt'
        ).path()
        ObjectDetector.labels = load_labels(labels_path)

    def _generator(self):
        if ObjectDetector.interpreter is None:
            ObjectDetector.load_model()
        if ObjectDetector.labels is None:
            ObjectDetector.load_labels()

        while True:
            # image preprocess
            image = self._get_input('camera')
            scale = set_input(
                    ObjectDetector.interpreter,
                    (image.shape[1], image.shape[0]),
                    lambda size: cv2.resize(image, size))

            # inference
            ObjectDetector.interpreter.invoke()
            objs = get_output(ObjectDetector.interpreter, 0.4, scale)

            # output
            draw_objects(image, objs, ObjectDetector.labels)
            yield image
