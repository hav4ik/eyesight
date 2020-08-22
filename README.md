# Eyesight

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Travis CI](https://api.travis-ci.com/hav4ik/eyesight.svg?branch=master)](https://travis-ci.com/github/hav4ik/eyesight/builds/)

Eyesight is a high-level minimalistic framework for Computer Vision on the Edge, written in Python. It was developed with focus on performance. Supports Coral Edge TPU, Pi Camera, and more.

# Installation

Currently, this package is not uploaded to PyPi. The easiest way to install it right now is:

    $ git clone https://github.com/hav4ik/eyesight.git
    $ cd eyesight/
    $ pip install -e .

This will install the package in your local environment (the best practice is to use either Conda or virtualenv to create an Python environment).

# Basics

The EyeSight framework allows developers to define asynchronuous computation graph for Computer Vision pipelines without
having to worry about thread management, timestamp synchronization, locking mechaisms and all that crap. The framework
consists of following elements:

* **Packages** is the thing that gets passed between computation nodes. A package can hold any type of python
  objects (images, numpy arrays, pandas dataframes, etc.) together with the full time-stamped history of its processing.
  For example, if an collection of bounding boxes was acquired from an object detection node, that took its input
  images from two cameras, then the package holding these boxes will include timestamps from both cameras and
  from the detection node, in chronological order.

* **Services** are the minimal computation unit in the EyeSight graph, represented as a node. It takes in outputs from
  input nodes (can be an image, numpy array, or anything) and feeds its output to child nodes (services).

* **Adapters** controls the input data stream into each **Service.** It decides the timestamp synchronization strategy
  for the input data.

* **Manager** manages the services assigned to it. Although **services** are self-sustained (e.g. if you turn off
  an input service, it will turn off all services that depends on its outputs), having a **service manager**
  is always handy.

Both **Services** and **Adapters** are implemented in lock-free and no-copy fashion, meaning the exposed variables are
thread-safe. However, sometimes they rely on the internal lock mechanism of Python, which is not always suitable for a
computer vision pipeline and it can significantly slow down the pipeline. For this reason, the framework by default
uses RW locks everywhere.

# Minimal Example

This is a minimal example of constructing an EyeSight computation graph. The full example can be found in `eyesight/examples/demo_basic.py`.

    import eyesight
    import eyesight.services as services


    # Raspberry Pi camera input node
    camera = services.PiCamera()

    # MobileNetV2 SSD COCO applied to the camera's outputs
    detector = services.ObjectDetector(camera)

    # Aside from detection, we also want to 
    tracker = services.OpticalFlowLucasKanade(camera)

    # Output visualization
    composer = services.DetectronDraw(
        image_stream = camera, detector=detector, tracker=tracker)

    # ServiceManager will automatically detect service's dependencies
    # and include them recursively as well
    manager = eyesight.ServiceManager(composer)

# Troubleshooting

  * **I'm using TensorFlow Lite and there's an FPS drop / memory leak.**
    This is probably not the problem of EyeSight, but the TensorFlow problem. Update TensorFlow to version 2.3.0.
    If you are using it with EdgeTPU, then any version of TensorFlow or tflite_runtime should work.
