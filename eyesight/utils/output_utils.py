import cv2
import numpy as np
import collections
from ..utils import backend_utils as backend


# Structure to hold objects
DetectionObject = collections.namedtuple(
        'DetectionObject', ['id', 'score', 'bbox'])


class BoundingBoxesNP:
    """Vectorized bounding boxes, harnessing the power of Numpy!
    """
    def __init__(self,
                 class_ids: np.ndarray,
                 xmin: np.ndarray,
                 ymin: np.ndarray,
                 xmax: np.ndarray,
                 ymax: np.ndarray,
                 scores: np.ndarray,
                 threshold: float = None):

        if backend._mode == 'debug':
            assert len(class_ids.shape) == 1
            assert len(xmin.shape) == 1
            assert len(ymin.shape) == 1
            assert len(xmax.shape) == 1
            assert len(ymax.shape) == 1
            assert len(scores.shape) == 1

            assert class_ids.shape[0] == xmin.shape[0]
            assert class_ids.shape[0] == ymin.shape[0]
            assert class_ids.shape[0] == xmax.shape[0]
            assert class_ids.shape[0] == ymax.shape[0]
            assert class_ids.shape[0] == scores.shape[0]

        self.class_ids = class_ids
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.scores = scores

        self.indices = np.arange(class_ids.shape[0])
        if threshold is not None:
            self.indices = self.indices[scores >= threshold]

    def __getitem__(self, i):
        indices = self.indices[i]
        return self.class_ids[indices], \
            self.xmin[indices], self.ymin[indices], \
            self.xmax[indices], self.ymax[indices], \
            self.scores[indices]

    def __len__(self):
        return self.indices.shape[0]

    def scale(self, scale_x, scale_y):
        self.xmin *= scale_x
        self.xmax *= scale_x
        self.ymin *= scale_y
        self.ymax *= scale_y
        return self


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
                  (bbox.xmin - 1, bbox.ymin - 1))


def draw_objects_np(img, objs, labels):
    """Draws the bounding box and label for each object."""
    for i in range(len(objs)):
        class_id, xmin, ymin, xmax, ymax, score = objs[i]
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                      (0, 255, 0), 2)
        text_over(img,
                  '{:s} ({:.1f}%)'.format(
                      labels.get(int(class_id), str(class_id)),
                      score * 100), (int(xmin) - 1, int(ymin) - 1))


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    indices = np.arange(256, dtype=np.uint8)

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
            is the color indexed by the corresponding element in the input
            label to the PASCAL color map.
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


def draw_tracking_sparse(img, prev_points, cur_points):
    """Draws tracking points for sparse (e.g. Lukas-Kanade) trackers
    """
    assert len(prev_points) == len(cur_points)
    for p0, p1 in zip(prev_points, cur_points):
        a, b = p0.ravel()
        c, d = p1.ravel()
        cv2.line(img, (a, b), (c, d), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(img, (c, d), 3, (0, 255, 0), -1)
