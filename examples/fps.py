import time
import datetime
import os

from imutils.video import FPS
from eyesight.utils.generic_utils import log
from eyesight import ServiceManager

if 'CAMERA' in os.environ:
    if os.environ['CAMERA'] == 'pi':
        from eyesight.services import PiCamera as Camera
    elif os.environ['CAMERA'] == 'opencv':
        from eyesight.services import CVCamera as Camera
    elif os.environ['CAMERA'] == 'img':
        from eyesight.services import ImageCamera as Camera
    elif os.environ['CAMERA'] == 'empty':
        from eyesight.services import EmptyCamera as Camera
    else:
        raise RuntimeError('Unknown CAMERA specified.')
else:
    from eyesight.services import EmptyCamera as Camera

if 'SERVICE' in os.environ:
    if os.environ['SERVICE'] == 'det':
        from eyesight.services import ObjectDetector as Service
    elif os.environ['SERVICE'] == 'sem':
        from eyesight.services import SemanticSegmentator as Service
    elif os.environ['SERVICE'] == 'draw':
        from eyesight.services import DetectronDraw as Service
    else:
        raise RuntimeError('Unknown SERVICE specified.')
else:
    from eyesight.services import EmptyService as Service


stream = Service(Camera())
manager = ServiceManager(stream)
manager.start()
log.info('Start measuring')

fps = FPS().start()
delta = 0.
while fps._numFrames < 200 and \
        (datetime.datetime.now() - fps._start).total_seconds() < 10:
    history, frame = stream.query()
    delta += time.time() - history[0].timestamp
    fps.update()
fps.stop()

log.info('Stopped measuring')
manager.stop()
delta /= fps._numFrames

log.info('Elapsed time: {:.2f}'.format(fps.elapsed()))
log.info('Approx. FPS: {:.2f}'.format(fps.fps()))
log.info('Average delay: {:.2f}'.format(delta))
