import cv2
import numpy as np
import time

from ..engine.base_service import BaseService
from ..engine.adapters import get as get_adapter
from ..utils.generic_utils import log


class OpticalFlowLucasKanade(BaseService):
    """
    Tracking by keypoints using Lukas-Kanade algorithm implemented in OpenCV.

    Args:
      camera: The camera service.
      renew_after: number of frames, after which the initial Shi-Tomashi
        points should be renewed.
      shitomashi_params: a dictionary of params for Shi-Tomashi good tracking
        points (corner) detector.
      lucas_kanade_params: a dictionary of params for Lucas-Kanade Optical
        Flow algorithm.
    """
    def __init__(self,
                 camera,
                 renew_after=30,
                 shitomashi_params=None,
                 lucas_kanade_params=None,
                 *args, **kwargs):

        super().__init__(
                adapter=get_adapter('simple')({'cam': camera}),
                *args, **kwargs)

        self._renew_after = renew_after
        self._shi_tomashi_params = shitomashi_params
        self._lucas_kanade_params = lucas_kanade_params

    def _generator(self):
        # params for ShiTomasi corner detection
        if self._shi_tomashi_params is None:
            feature_params = dict(
                    maxCorners=100,
                    qualityLevel=0.03,
                    minDistance=7,
                    blockSize=7)
        else:
            feature_params = self._shi_tomashi_params

        # Parameters for lucas kanade optical flow
        if self._lucas_kanade_params is None:
            lk_params = dict(
                    winSize=(15, 15),
                    maxLevel=2,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        10, 0.03))
        else:
            lk_params = self._lucas_kanade_params

        prev_img = None
        p0 = None
        elapsed_frames = 0
        while True:
            image = self._get_inputs('cam')
            assert isinstance(image, np.ndarray)
            assert len(image.shape) == 2 or (
                    len(image.shape) == 3 and image.shape[2] == 3)

            # Get image from camera and convert it to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Prepare first image if needed
            if prev_img is None:
                prev_img = image
                yield None
                begin = time.time()
                p0 = cv2.goodFeaturesToTrack(
                        image, mask=None, **feature_params)
                log.debug('Found {} features to track in {} sec'.format(
                        len(p0), time.time() - begin))
                elapsed_frames = 0
                continue

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_img, image, p0, None, **lk_params)

            # Select and yield good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            yield (good_old, good_new)

            # Now update the previous frame and previous points
            prev_img = image
            p0 = good_new.reshape(-1, 1, 2)

            # Re-initialize tracking points if needed
            elapsed_frames += 1
            if elapsed_frames > self._renew_after:
                begin = time.time()
                p0 = cv2.goodFeaturesToTrack(
                        image, mask=None, **feature_params)
                log.debug('Found {} features to track in {} sec'.format(
                        len(p0), time.time() - begin))
                elapsed_frames = 0


class OpticalFlowFarneback(BaseService):

    def __init__(self, camera, *args, **kwargs):
        super().__init__(
                adapter=get_adapter('simple')({'cam': camera}),
                *args, **kwargs)

    def _generator(self):
        prev_img = None
        while True:
            image = self._get_inputs('cam')
            assert isinstance(image, np.ndarray)
            assert len(image.shape) == 2 or (
                    len(image.shape) == 3 and image.shape[2] == 3)

            # Get image from camera and convert it to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Prepare first image if needed
            if prev_img is None:
                prev_img = image
                yield None

            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                    prev_img, image, flow=None,
                    pyr_scale=0.5, levels=2, winsize=15, iterations=1,
                    poly_n=3, poly_sigma=1.2, flags=0)

            yield flow
            prev_img = image
