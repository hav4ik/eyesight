import time
import numpy as np
from pai.services import BaseStreamService

# If we can import picamera, i.e. we are on a Raspberry Pi
PLATFORM_RPI = True
try:
    import picamera
    import picamera.array
except:
    PLATFORM_RPI = False

# If OpenCV is available in current system
HAS_OPENCV = True
try:
    import cv2
except:
    HAS_OPENCV = False


class DefaultCamera(BaseStreamService):
    """Just for basic testing"""
    @staticmethod
    def frames():
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        while True:
            img = 255 - img
            yield img


class PiCamera(BaseStreamService):
    """Pi Camera service"""
    def __init__(self):
        if not PLATFORM_RPI:
            raise NotImplementedError('Only available for Raspberry Pi')
        super().__init__()

    @staticmethod
    def frames():
        with picamera.PiCamera() as camera:
            shape = (640, 480)
            camera.resolution = shape
            camera.framerate = 32
            camera.sensor_mode = 1
            raw_capture = picamera.array.PiRGBArray(camera, size=shape)

            # let the camera warm up
            time.sleep(2)

            for _ in camera.capture_continuous(
                    raw_capture, format='bgr', use_video_port=True):
                yield raw_capture.array
                raw_capture.truncate(0)


class CVCamera(BaseStreamService):
    """WebCam service"""
    video_source = 0

    def __init__(self):
        if not HAS_OPENCV:
            raise NotImplementedError('OpenCV required')

        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            CVCamera.video_source = \
                    int(os.environ['OPENCV_CAMERA_SOURCE'])
        super().__init__()

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(CVCamera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            ret, img = camera.read()
            if not ret:
                raise RuntimeError('Next frames are not available.')
                break
            yield img
