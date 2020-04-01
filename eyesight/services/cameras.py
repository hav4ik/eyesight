import os
import time
import numpy as np
from eyesight import BaseService

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


class DefaultCamera(BaseService):
    """Just for basic testing"""

    def __init__(self):
        super().__init__()

    def _generator(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        while True:
            img = 255 - img
            yield img


class PiCamera(BaseService):
    """Pi Camera service"""

    def __init__(self,
                 resolution=(640, 480),
                 framerate=32,
                 sensor_mode=1,
                 flipud=False):

        if not PLATFORM_RPI:
            raise NotImplementedError('Only available for Raspberry Pi')

        self.resolution = resolution
        self.framerate = framerate
        self.sensor_mode = sensor_mode
        super().__init__()

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
                    raw_capture, format='bgr', use_video_port=True):
                yield raw_capture.array
                raw_capture.truncate(0)


class CVCamera(BaseService):
    """WebCam service"""

    def __init__(self, video_source=0):
        if not HAS_OPENCV:
            raise NotImplementedError('OpenCV required')

        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            CVCamera.video_source = \
                    int(os.environ['OPENCV_CAMERA_SOURCE'])

        self.video_source = video_source
        super().__init__()

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
