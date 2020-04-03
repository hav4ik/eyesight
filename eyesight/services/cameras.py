import os
import time
import numpy as np

from ..engine.base_service import BaseService
from ..utils import Resource


# If we can import picamera, i.e. we are on a Raspberry Pi
PLATFORM_RPI = True
try:
    import picamera
    import picamera.array
except ImportError:
    PLATFORM_RPI = False

# If OpenCV is available in current system
HAS_OPENCV = True
try:
    import cv2
except ImportError:
    HAS_OPENCV = False


class EmptyCamera(BaseService):
    """Just for basic testing
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generator(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        while True:
            img = 255 - img
            yield img


class ImageCamera(BaseService):
    """To test computer vision algorithms
    """
    def __init__(self, images=None, size=(640, 480), *args, **kwargs):
        if not HAS_OPENCV:
            raise NotImplementedError('OpenCV required')
        if images is None:
            images = [Resource(
                collection_name='test_images',
                url='https://photolemur.com/uploads/blog/'
                    'thierry-ambraisse-200857.jpg').path()]
        self.size = size

        self.images = []
        if isinstance(images, str):
            self.images = self._get_image_or_imdir(
                    os.path.expanduser(images))
        elif isinstance(images, list):
            for path in images:
                self.images.extend(self._get_image_or_imdir(
                    os.path.expanduser(path)))
        super().__init__(*args, **kwargs)

    def _generator(self):
        while True:
            for path in self.images:
                img = cv2.resize(cv2.imread(path), self.size)
                yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _get_image_or_imdir(path):
        if os.path.isdir(path):
            return [os.path.join(path, x) for x in os.listdir(path)]
        elif os.path.isfile(path):
            return [path]
        else:
            raise RuntimeError('Path {} does not exist.'.format(path))


class PiCamera(BaseService):
    """Pi Camera service"""

    def __init__(self,
                 resolution=(640, 480),
                 framerate=32,
                 sensor_mode=1,
                 flipud=False,
                 *args,
                 **kwargs):

        if not PLATFORM_RPI:
            raise NotImplementedError('Only available for Raspberry Pi')

        self.resolution = resolution
        self.framerate = framerate
        self.sensor_mode = sensor_mode
        super().__init__(*args, **kwargs)

    def _generator(self):
        with picamera.PiCamera() as camera:
            camera.resolution = self.resolution
            camera.framerate = self.framerate
            camera.sensor_mode = self.sensor_mode
            raw_capture = picamera.array.PiRGBArray(
                    camera, size=self.resolution)

            # let the camera warm up
            time.sleep(2)

            for _ in camera.capture_continuous(
                    raw_capture, format='rgb', use_video_port=True):
                yield raw_capture.array
                raw_capture.truncate(0)


class CVCamera(BaseService):
    """WebCam service"""

    def __init__(self, video_source=0, *args, **kwargs):
        if not HAS_OPENCV:
            raise NotImplementedError('OpenCV required')

        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            CVCamera.video_source = \
                    int(os.environ['OPENCV_CAMERA_SOURCE'])

        self.video_source = video_source
        super().__init__(*args, **kwargs)

    def _generator(self):
        camera = cv2.VideoCapture(self.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            ret, img = camera.read()
            if not ret:
                raise RuntimeError('Next frames are not available.')
                break
            yield img
