import datetime
import time
import os
from imutils.video import FPS
from pai.utils import Log

if 'CAMERA' in os.environ:
    if os.environ['CAMERA'] == 'pi':
        from pai.services import PiCamera as Camera
    elif os.environ['CAMERA'] == 'opencv':
        from pai.services import CVCamera as Camera
    else:
        raise RuntimeError('Unknown CAMERA specified.')
else:
    from pai.services import DefaultCamera as Camera

if 'SERVICE' in os.environ:
    if os.environ['SERVICE'] == 'det':
        from pai.services import ObjectDetector as Service
    else:
        raise RuntimeError('Unknown SERVICE specified.')
else:
    Service = lambda x: x


stream = Service(Camera())
Log.info('Start measuring')

fps = FPS().start()
while fps._numFrames < 200 and \
        (datetime.datetime.now() - fps._start).total_seconds() < 10:
    t, frame = stream.get_frame()
    fps.update()
fps.stop()

Log.info('Elapsed time: {:.2f}'.format(fps.elapsed()))
Log.info('Approx. FPS: {:.2f}'.format(fps.fps()))

