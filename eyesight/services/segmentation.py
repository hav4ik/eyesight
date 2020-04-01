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


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.
  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  indices = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((indices >> channel) & 1) << shift
    indices >>= 3

  return colormap

def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.
  Args:
    label: A 2D array with integer type, storing the segmentation label.
  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.
  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


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
                        EDGETPU_SHARED_LIB,
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
    interpreter = None

    def __init__(self, camera, **kwargs):
        super().__init__(input_services={'camera': camera}, **kwargs)

    @staticmethod
    def load_model():
        model_path = Resource(
                collection_name='mobilenet_v2_deeplab_v3_pascal2012',
                url='https://github.com/google-coral/'
                    'edgetpu/raw/master/test_data/'
                    'deeplabv3_mnv2_pascal_quant_edgetpu.tflite'
        ).path()
        SemanticSegmentator.interpreter = make_interpreter(model_path)
        SemanticSegmentator.interpreter.allocate_tensors()

    def _generator(self):
        if SemanticSegmentator.interpreter is None:
            SemanticSegmentator.load_model()

        while True:
            image = self._get_input('camera')
            scale, img  = set_input(
                    SemanticSegmentator.interpreter,
                    (image.shape[1], image.shape[0]),
                    lambda size: cv2.resize(image, size))

            SemanticSegmentator.interpreter.invoke()
            result = output_tensor(SemanticSegmentator.interpreter, 0)
            result = result.copy()[
                    :int(image.shape[0] * scale[0]),
                    :int(image.shape[1] * scale[1])]

            vis_res = label_to_color_image(
                    result.astype(np.int)).astype(np.uint8)
            vis_res = 2 * (vis_res // 3) + img // 3
            yield vis_res

