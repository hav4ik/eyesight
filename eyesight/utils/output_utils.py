import cv2
import numpy as np
import collections


# Structure to hold objects
DetectionObject = collections.namedtuple(
        'DetectionObject', ['id', 'score', 'bbox'])


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
        cv2.circle(img, (c, d), 2, (127, 255, 127), -1)
        cv2.circle(img, (c, d), 2, (0, 255, 0), -1)
