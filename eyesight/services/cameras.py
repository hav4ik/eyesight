import os
import time
import numpy as np

from ..engine.base_service import BaseService
from ..utils.generic_utils import Resource
from ..utils import backend_utils as backend
from ..utils.generic_utils import log

import cv2
if backend._USING_RASPBERRYPI_CAMERA:
    import picamera
    import picamera.array


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
    def __init__(self,
                 images=None,
                 cache_images=True,
                 size=(640, 480),
                 *args, **kwargs):

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

        self.cache_images = cache_images
        super().__init__(*args, **kwargs)

    def _generator(self):
        if self.cache_images:
            image_cache = []
            for path in self.images:
                img = cv2.resize(cv2.imread(path), self.size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_cache.append(img)

        while True:
            if not self.cache_images:
                for path in self.images:
                    img = cv2.resize(cv2.imread(path), self.size)
                    yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                for img in image_cache:
                    yield img

    @staticmethod
    def _get_image_or_imdir(path):
        if os.path.isdir(path):
            return [os.path.join(path, x) for x in os.listdir(path)]
        elif os.path.isfile(path):
            return [path]
        else:
            raise RuntimeError('Path {} does not exist.'.format(path))


class VideoFileReader(BaseService):
    def __init__(self,
                 video_path=None,
                 size=(640, 480),
                 cache_all=True,
                 target_fps=200,
                 *args, **kwargs):
        if video_path is None:
            self.video_path = Resource(
                    collection_name='test_videos',
                    url='https://github.com/hav4ik/static-files/raw/'
                        'master/eyesight/street_vid.mp4').path()
        else:
            self.video_path = video_path
        self.size = size
        self.cache_all = cache_all
        self.target_fps = target_fps
        super().__init__(*args, **kwargs)

    def _generator(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_counter = 0

        if self.cache_all:
            frame_cache = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(
                        cv2.resize(frame, self.size), cv2.COLOR_BGR2RGB)
                frame_cache.append(frame)

            if len(frame_cache) == 0:
                raise RuntimeError('0 frames in video cache.')
            cap.release()

        if self.target_fps is not None:
            prev_iteration = time.time()

        while True:
            if self.cache_all:
                yield frame_cache[frame_counter]
                frame_counter = (frame_counter + 1) % len(frame_cache)
            else:
                ret, frame = cap.read()
                frame_counter += 1
                if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    frame_counter = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                if not ret:
                    log.error('Unable to read a frame from video.')
                    yield None
                else:
                    yield cv2.cvtColor(
                            cv2.resize(frame, self.size), cv2.COLOR_BGR2RGB)

            if self.target_fps is not None:
                now = time.time()
                diff = now - prev_iteration
                prev_iteration = now
                target_delay = 1. / self.target_fps
                if diff < target_delay:
                    time.sleep(target_delay - diff)


class PiCamera(BaseService):
    """Pi Camera service"""

    def __init__(self,
                 resolution=(640, 480),
                 framerate=32,
                 sensor_mode=1,
                 flipud=True,
                 *args,
                 **kwargs):

        if not backend._USING_RASPBERRYPI_CAMERA:
            raise NotImplementedError(
                    'Only available for Raspberry Pi - missing '
                    '`picamera` module. Install it with `pip install '
                    'picamera` and make sure you have a PiCamera '
                    'attached.')

        self.resolution = resolution
        self.framerate = framerate
        self.sensor_mode = sensor_mode
        self.flipud = flipud
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

                # we try to keep the copies at minimum amount, so if
                # self._no_copy is True, we don't need to make a copy
                # right here.
                if self.flipud:
                    if self._no_copy:
                        yield np.flipud(raw_capture.array).copy()
                    else:
                        yield np.flipud(raw_capture.array)
                else:
                    if self._no_copy:
                        yield raw_capture.array.copy()
                    else:
                        yield raw_capture.array
                raw_capture.truncate(0)


class CVCamera(BaseService):
    """WebCam service"""

    def __init__(self, video_source=0, *args, **kwargs):
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
            try:
                ret, img = camera.read()
            except:
                camera.release()
                camera = cv2.VideoCapture(self.video_source)
                ret, img = camera.read()

            if not ret:
                raise RuntimeError('Next frames are not available.')
                break
            yield img

        camera.release()
