from collections import deque
from collections import namedtuple
import numpy as np
import cv2

from ..engine.base_service import BaseService
from ..engine.adapters import get as get_adapter
from ..utils.generic_utils import log
from ..utils.output_utils import draw_objects
from ..utils.output_utils import label_to_color_image


class EmptyService(BaseService):
    """
    A service that does nothing and just passes the first input through
    itself. Used for testing other modules and services.
    """
    def __init__(self, service, *args, **kwargs):
        super().__init__(
                adapter=get_adapter('simple')([service]),
                *args, **kwargs)

    def _generator(self):
        while True:
            yield self._get_inputs(0)


class PerformanceBar(BaseService):
    """Writes performance statistics info on a bar on top of received image.

    Following statistics are calculated and displayed:
      * FPS (Frames Per Second)
      * Delay (time difference between when the frames were received and
            when it is last processed.

    # Arguments
        service: a BaseService object that returns an 3-channel image
        n_frames: how much frames should be taken into account when the
            performance is calculated. If set to 0, all frames would be
            used.
    """
    BacklogItem = namedtuple('BacklogItem', 'timestamp delay')

    def __init__(self, service, n_frames=200, *args, **kwargs):
        self.n_frames = n_frames
        super().__init__(
                adapter=get_adapter('simple')({'service': service}),
                *args, **kwargs)

    def _generator(self):
        total_delay = 0.
        frame_counter = 0
        start_time = None

        if self.n_frames > 0:
            performance_log = deque()

        while True:
            # After `_get_inputs`, we will have a full history tape of this
            # package for analyze (the last entry will be this class).
            image = self._get_inputs('service')
            assert isinstance(image, np.ndarray) and len(image.shape) == 3 \
                and image.shape[2] == 3 and image.dtype == np.uint8

            # Delay between the timestamp of first introduction of the
            # frame and the timestamp of when it's processed the last
            delay = self._history_tape[-1].timestamp - \
                self._history_tape[0].timestamp

            # Accumulating statistics
            if self.n_frames > 0:
                performance_log.append(PerformanceBar.BacklogItem(
                    self._history_tape[-1].timestamp, delay))
                if len(performance_log) > self.n_frames:
                    el = performance_log.popleft()
                    total_delay -= el.delay
                if len(performance_log) == 1:
                    continue
            else:
                frame_counter += 1
                if start_time is None:
                    start_time = self._history_tape[-1].timestamp
                    continue
            total_delay += delay

            # Calculating performance statistics
            if self.n_frames > 0:
                lst = list(performance_log)
                avg_fps = len(performance_log) / (
                        lst[-1].timestamp - lst[0].timestamp)
                avg_delay = total_delay / len(performance_log)
            else:
                now = self._history_tape[-1].timestamp
                avg_fps = float(frame_counter) / (now - start_time)
                avg_delay = total_delay / frame_counter

            # Draw a status bar on top of the image
            tx_color = (255, 255, 255)
            bg_color = (0, 0, 0)
            thickness = 1
            padding = (5, 5)
            font_scale = 0.6
            font = cv2.FONT_HERSHEY_DUPLEX

            text = 'FPS: {:.1f} | Delay: {:.1f}ms'.format(
                    avg_fps, avg_delay * 1000.)
            (tw, th), baseline = cv2.getTextSize(
                    text, font, fontScale=font_scale, thickness=thickness)
            if tw > image.shape[1]:
                log.warning(
                    'Status bar is too wide for image ({} > {})'.format(
                        tw, image.shape[1]))

            xmin, ymax = padding[0], th + 2 * padding[1]
            display_img = np.empty(
                    shape=(
                        image.shape[0] + th + 2 * padding[1],
                        image.shape[1], 3),
                    dtype=np.uint8)
            display_img[:th + 2 * padding[1], :, :] = np.array(bg_color)
            display_img[th + 2 * padding[1]:, :, :] = image
            cv2.putText(display_img, text,
                        (xmin + padding[0], ymax - padding[1]),
                        font, fontScale=font_scale,
                        color=tx_color, thickness=thickness,
                        lineType=cv2.LINE_AA)

            # Return the image with status bar on top
            yield display_img


class DetectronDraw(BaseService):
    """
    Aggregation and drawing of various vision services. Supported services
    and required output formats:

      - Object Detectors. Should output a tuple of two elements: a list of
            `DetectionObject`, defined in `utils/output_utils.py`, and the
            labels (a list of names for each id).
      - Image Segmentators. Should output a color map together with labels
            (a dict of names for each color).

    # Arguments:
        image_stream: the service that feeds images to all computer vision
            services in the pipeline. Usually it's the camera service.
        detector: (optional) object detection service, output format of it
            should follow the format described above.
        segmentator: (optional) semantic segmentation service, its output
            format should follow the format described above.
    """
    def __init__(self,
                 image_stream,
                 detector=None,
                 segmentator=None,
                 tracker=None):

        self._has_detector = False
        self._has_segmentator = False
        self._has_tracker = False
        input_services = {'image_stream': image_stream}

        if detector is not None:
            self._has_detector = True
            input_services['detector'] = detector

        if segmentator is not None:
            self._has_segmentator = True
            input_services['segmentator'] = segmentator

        if tracker is not None:
            self._has_tracker = True
            input_services['tracker'] = tracker

        super().__init__(adapter=get_adapter('sync')(input_services))

    def _generator(self):
        while True:
            image, detections, segmentation, tracking = self._get_inputs(
                    'image_stream', 'detector', 'segmentator', 'tracker')

            if image is None:
                log.warning(
                        '`image_stream` yielded None (expected behavior), '
                        'continue.')
                continue
            assert isinstance(image, np.ndarray) and len(image.shape) == 3 \
                and image.shape[2] == 3 and image.dtype == np.uint8

            if self._has_segmentator and segmentation is not None:
                assert isinstance(segmentation, np.ndarray) and \
                    len(segmentation.shape) == 2

                vis_res = label_to_color_image(
                        segmentation.astype(np.int)).astype(np.uint8)
                if vis_res.shape != image.shape:
                    vis_res = cv2.resize(vis_res, image.shape)
                image = 2 * (vis_res // 3) + image // 3

            if self._has_detector and detections is not None:
                objects, labels = detections
                draw_objects(image, objects, labels)

            yield image
