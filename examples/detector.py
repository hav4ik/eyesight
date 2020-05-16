#!/usr/bin/env python
import os

from flask import Flask, render_template, Response
import cv2

from eyesight.services import PerformanceBar, DetectronDraw
from eyesight import ServiceManager

if 'CAMERA' in os.environ:
    if os.environ['CAMERA'] == 'pi':
        from eyesight.services import PiCamera as Camera
    elif os.environ['CAMERA'] == 'opencv':
        from eyesight.services import CVCamera as Camera
    elif os.environ['CAMERA'] == 'img':
        from eyesight.services import ImageCamera as Camera
    elif os.environ['CAMERA'] == 'vid':
        from eyesight.services import VideoFileReader as Camera
    else:
        raise RuntimeError('Unknown CAMERA specified.')
else:
    from eyesight.services import EmptyCamera as Camera

if 'SERVICE' in os.environ:
    if os.environ['SERVICE'] == 'det':
        from eyesight.services import ObjectDetector as Service
    elif os.environ['SERVICE'] == 'sem':
        from eyesight.services import SemanticSegmentator as Service
    elif os.environ['SERVICE'] == 'track':
        from eyesight.services import OpticalFlowLucasKanade as Service
    else:
        raise RuntimeError('Unknown SERVICE specified.')
else:
    from eyesight.services import EmptyService as Service


app = Flask(__name__)

raspberry_cam = Camera()
selected_service = Service(raspberry_cam)
if 'SERVICE' in os.environ:
    if os.environ['SERVICE'] == 'det':
        output_drawer = DetectronDraw(
                image_stream=raspberry_cam, detector=selected_service)
    elif os.environ['SERVICE'] == 'sem':
        output_drawer = DetectronDraw(
                image_stream=raspberry_cam, segmentator=selected_service)
    elif os.environ['SERVICE'] == 'track':
        output_drawer = DetectronDraw(
                image_stream=raspberry_cam, tracker=selected_service)
else:
    output_drawer = DetectronDraw(image_stream=raspberry_cam)

service = PerformanceBar(output_drawer)
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
