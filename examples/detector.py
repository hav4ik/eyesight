#!/usr/bin/env python
import os

from flask import Flask, render_template, Response
import cv2

from eyesight.services import PerformanceBar
from eyesight import ServiceManager

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
    from eyesight.services import EmptyCamera as Camera

if 'SERVICE' in os.environ:
    if os.environ['SERVICE'] == 'det':
        from eyesight.services import ObjectDetector as Service
    elif os.environ['SERVICE'] == 'sem':
        from eyesight.services import SemanticSegmentator as Service
    else:
        raise RuntimeError('Unknown SERVICE specified.')
else:
    from eyesight.services import EmptyService as Service


app = Flask(__name__)
service = PerformanceBar(Service(Camera()))
manager = ServiceManager(service)
manager.start()


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = cv2.cvtColor(camera.query()[1], cv2.COLOR_RGB2BGR)
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(service),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
