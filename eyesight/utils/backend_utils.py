import os
import yaml
import platform

from .generic_utils import log


# Set Eyesight base dir path given EYESIGHT_HOME env variable, if applicable.
# Otherwise either ~/.eyesight or /tmp. Then, create the dir if not exists.
_eyesight_base_dir = os.path.expanduser('~')
if not os.access(_eyesight_base_dir, os.W_OK):
    _eyesight_base_dir = '/tmp'
_eyesight_dir = os.path.join(_eyesight_base_dir, '.eyesight')
_eyesight_dir = os.environ.get('EYESIGHT_HOME', _eyesight_dir)
if not os.path.isdir(_eyesight_dir):
    os.makedirs(_eyesight_dir)
log.debug('[Init] Eyesight dir: {}'.format(_eyesight_dir))

# Default neural network inference backend: TF-Lite
_backend = 'tflite'

# There are 2 modes: 'debug' and 'ship'
_mode = 'debug'

# Additional information about available libraries and modules, all values
# defaults to `False` or `None`.
_USING_TFLITE_RUNTIME = False
_USING_TENSORFLOW_TFLITE = False
_USING_EDGE_TPU = False
_USING_RASPBERRYPI_CAMERA = False
_EDGETPU_SHARED_LIB = None

# Attempt to read from Eyesight config file. Otherwise, try to set it based
# on EYESIGHT_BACKEND environment variable.
_config_path = os.path.expanduser(
        os.path.join(_eyesight_dir, 'eyesight.yaml'))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as stream:
            _config = yaml.safe_load(stream)
    except ValueError:
        _config = dict()
    _backend = _config.get('backend', _backend)
    _mode = _config.get('mode', _mode)
else:
    _backend = os.environ.get('EYESIGHT_BACKEND', _backend)
    _mode = os.environ.get('EYESIGHT_MODE', _mode)
log.debug('[Init] Backend is set to {}'.format(_backend))
log.debug('[Init] Mode is set to {}'.format(_mode))

# If successfully imported, one can use it as `backend.tflite`.
if _backend == 'tflite':
    # Try to import tflite_runtime if available. Otherwise, try to import
    # it from standard tensorflow.
    try:
        import tflite_runtime.interpreter as tflite
        _USING_TFLITE_RUNTIME = True
        log.debug('[Init] Using `tflite` from `tflite_runtime`.')
    except ImportError:
        try:
            import tensorflow.lite as tflite
            _USING_TENSORFLOW_TFLITE = True
            log.debug('[Init] Using `tflite` from `tensorflow`.')
        except ImportError as e:
            raise ImportError(
                'Backend "tflite" not found. Please make sure that either'
                'tensorflow or tflite_runtime is installed') from e

    _EDGETPU_SHARED_LIB = {
        'Linux': 'libedgetpu.so.1',
        'Darwin': 'libedgetpu.1.dylib',
        'Windows': 'edgetpu.dll'
    }[platform.system()]

    try:
        if _USING_TENSORFLOW_TFLITE:
            tflite.experimental.load_delegate(_EDGETPU_SHARED_LIB)
        elif _USING_TFLITE_RUNTIME:
            tflite.load_delegate(_EDGETPU_SHARED_LIB)
        _USING_EDGE_TPU = True
        log.debug('[Init] Using Google Coral Edge TPU.')
    except (ValueError, RuntimeError):
        _EDGETPU_SHARED_LIB = None

else:
    raise ValueError('Backend `{}` not supported.'.format(_backend))

# If we can import picamera, i.e. we are on a Raspberry Pi
try:
    import picamera
    log.debug('[Init] Found picamera module.')
    _USING_RASPBERRYPI_CAMERA = True
except ImportError:
    _USING_RASPBERRYPI_CAMERA = False
