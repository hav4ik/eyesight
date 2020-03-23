#!/usr/bin/env python
import os

from flask import Flask, render_template, Response
import cv2

from pai.services import ObjectDetector
if 'CAMERA' in os.environ:
    if os.environ['CAMERA'] == 'pi':
        from pai.services import PiCamera as Camera
    elif os.environ['CAMERA'] == 'opencv':
        from pai.services import CVCamera as Camera
    else:
        raise RuntimeError('Unknown CAMERA specified.')
else:
    from pai.services import DefaultCamera as Camera


app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = cv2.imencode('.jpg', camera.get_frame()[1])[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(ObjectDetector(Camera())),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
