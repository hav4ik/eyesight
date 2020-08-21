import os
import cv2
import numpy as np

from ..engine.base_service import BaseService
from ..engine.adapters import get as get_adapter
from ..utils.generic_utils import Resource
from ..utils.output_utils import BoundingBoxesNP
from ..utils import backend_utils as backend

if backend._USING_TENSORFLOW_TFLITE:
    import tensorflow.lite as tflite
elif backend._USING_TFLITE_RUNTIME:
    import tflite_runtime.interpreter as tflite


def output_tensor(interpreter, i):
    """Returns output tensor view."""
    tensor = interpreter.tensor(interpreter.get_output_details()[i]['index'])()
    return np.squeeze(tensor)


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


class ObjectDetector(BaseService):
    """Object detection service, based on MobileNet V2 SSD Coco
    """
    def __init__(self, camera, *args, **kwargs):
        if not (backend._USING_TFLITE_RUNTIME or
                backend._USING_TENSORFLOW_TFLITE):
            raise NotImplementedError(
                    'TensorflowLite not found. Install it with '
                    '`pip install tflite_runtime`.')

        self.interpreter = None
        self.labels = None
        self.cam_width = None
        self.cam_height = None

        super().__init__(
                adapter=get_adapter('simple')({'cam': camera}),
                *args, **kwargs)

    def load_model(self):
        """Load the model and allocate memory for it
        """
        if backend._USING_EDGE_TPU:
            model_path = Resource(
                collection_name='mobilenet_ssd_v2_coco',
                url='https://github.com/google-coral/'
                    'edgetpu/raw/master/test_data/'
                    'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
            ).path()
        else:
            archive_path = Resource(
                collection_name='mobilenet_ssd_v1_coco',
                url='https://storage.googleapis.com/'
                    'download.tensorflow.org/models/tflite/'
                    'coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip',
                is_archive=True,
            ).path()
            model_path = os.path.join(archive_path, 'detect.tflite')

        # Create an interpreter and allocate memory for it
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()

        # Get input shape
        _, height, width, _ = self.interpreter.get_input_details()[0]['shape']
        self.input_height = height
        self.input_width = width

        # Get index reference to input tensor
        self.input_tensor_index = \
            self.interpreter.get_input_details()[0]['index']

        # Get reference to output tensors
        self.boxes_index = \
            self.interpreter.get_output_details()[0]['index']
        self.class_ids_index = \
            self.interpreter.get_output_details()[1]['index']
        self.scores_index = \
            self.interpreter.get_output_details()[2]['index']
        self.count_index = \
            self.interpreter.get_output_details()[3]['index']

    def set_interpreter_input(self, image):
        """Copies a resized and properly zero-padded image to the input tensor.
        """
        w, h = image.shape[1], image.shape[0]
        scale_w, scale_h = self.input_width / w, self.input_height / h

        tensor = self.interpreter.tensor(self.input_tensor_index)()[0]
        cv2.resize(
                image,
                (self.input_width, self.input_height),
                dst=tensor[:h, :w],
                interpolation=cv2.INTER_NEAREST)
        return scale_w, scale_h

    def get_interpreter_output(
            self, score_threshold=0.4, image_scale=(1., 1.)):
        image_scale_x, image_scale_y = image_scale
        sx = self.input_width / image_scale_x
        sy = self.input_height / image_scale_y

        boxes = np.squeeze(
                self.interpreter.tensor(self.boxes_index)())
        class_ids = np.squeeze(
                self.interpreter.tensor(self.class_ids_index)())
        scores = np.squeeze(
                self.interpreter.tensor(self.scores_index)())
        count = int(np.squeeze(
            self.interpreter.tensor(self.count_index)()))

        return BoundingBoxesNP(
                class_ids[:count].copy(),
                boxes[:count, 1].copy(), boxes[:count, 0].copy(),
                boxes[:count, 3].copy(), boxes[:count, 2].copy(),
                scores[:count].copy(),
                score_threshold
            ).scale(sx, sy)

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
            scale_w, scale_h = self.set_interpreter_input(image)

            # inference
            self.interpreter.invoke()

            # output
            objs = self.get_interpreter_output(0.4, (scale_w, scale_h))
            yield objs, self.labels
