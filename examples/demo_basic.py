from flask import Flask, render_template, Response
import cv2

import eyesight
import eyesight.services as services


def construct_eyesight_pipeline():
    # First, get camera
    camera = services.PiCamera()
    # camera = services.VideoFileReader()

    # Define services
    detector = services.ObjectDetector(camera)
    segmentator = services.SemanticSegmentator(camera)
    tracker = services.OpticalFlowLucasKanade(camera)
    contour_extrapolator = services.SegmentationExtrapolator(
            segmentator=segmentator, tracker=tracker, orig_size=(640, 480))

    # Output services
    composer = services.DetectronDraw(
        image_stream=camera,
        # detector=detector,
        # segmentator=segmentator,
        tracker=tracker,
        contours=contour_extrapolator,
    )
    return services.PerformanceBar(composer)


# Flask App
app = Flask(__name__)

# Define and start eyesight pipeline
service = construct_eyesight_pipeline()
manager = eyesight.ServiceManager(service)
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
