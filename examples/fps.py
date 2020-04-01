import time
import datetime
import os

from imutils.video import FPS
from eyesight.utils import log

if 'CAMERA' in os.environ:
    if os.environ['CAMERA'] == 'pi':
        from eyesight.services import PiCamera as Camera
    elif os.environ['CAMERA'] == 'opencv':
        from eyesight.services import CVCamera as Camera
    elif os.environ['CAMERA'] == 'img':
        from eyesight.services import ImageCamera as Camera
    else:
        raise RuntimeError('Unknown CAMERA specified.')
else:
    from eyesight.services import DefaultCamera as Camera

if 'SERVICE' in os.environ:
    if os.environ['SERVICE'] == 'det':
        from eyesight.services import ObjectDetector as Service
    else:
        raise RuntimeError('Unknown SERVICE specified.')
else:
    def Service(x): return x


stream = Service(Camera())
stream.start()
log.info('Start measuring')

fps = FPS().start()
delta = 0.
while fps._numFrames < 200 and \
        (datetime.datetime.now() - fps._start).total_seconds() < 10:
    history, frame = stream.query()
    delta += time.time() - history[0].timestamp
    fps.update()
fps.stop()
delta /= fps._numFrames

log.info('Elapsed time: {:.2f}'.format(fps.elapsed()))
log.info('Approx. FPS: {:.2f}'.format(fps.fps()))
log.info('Average delay: {:.2f}'.format(delta))
