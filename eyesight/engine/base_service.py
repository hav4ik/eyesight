import time
import threading
from copy import deepcopy
from abc import ABCMeta, abstractmethod

from readerwriterlock import rwlock
from ..utils.generic_utils import log
from ..utils.generic_utils import class_name
from .adapters import get as get_adapter


class ClientEventHandler:
    """
    An Event-like class that signals all active clients when a new frame
    is available. Basically holds a pair (Event, timestamp) for each client
    and invokes `wait` and `set` for each of them.
    """
    def __init__(self, inactivity_timeout=5):
        self.events = dict()
        self.inactivity_timeout = inactivity_timeout

    def wait(self, timeout=None):
        """Invoked from each client's thread to wait for the next frame"""
        ident = threading.get_ident()
        if ident not in self.events:
            self.events[ident] = [threading.Event(), time.time()]
            return True
        return self.events[ident][0].wait(timeout)

    def set(self):
        """
        Invoked by the camera thread when a new frame is available. If
        the event is `Set`, then client has already processed the previous
        frame. Otherwise, the client is considered inactive. If a client
        is inactive for more than 5 seconds, he will be removed.
        """
        now = time.time()
        inactive_clients = []
        for ident, event in self.events.items():
            if not event[0].isSet():
                event[0].set()
                event[1] = now
            else:
                if now - event[1] > self.inactivity_timeout:
                    inactive_clients.append(ident)

        for ident in inactive_clients:
            self.events.pop(ident, None)

    def clear(self):
        """Invoked from each client's thread after a frame is processed"""
        self.events[threading.get_ident()][0].clear()

    def last(self):
        """Time (Unix timestamp) of the last image received by client"""
        return self.events[threading.get_ident()][1]


class HistoryItem:
    """History of each package when it's processed by a service.
    """
    def __init__(self, timestamp, invoker):
        self.timestamp = timestamp
        self.invoker = invoker

    def __str__(self):
        return 'HistoryItem({}, {})'.format(str(self.timestamp), self.invoker)


class BaseService(metaclass=ABCMeta):
    """Base class for each service.

    # Arguments
        input_services: a dict of service instances that may or may not
            have been started.
    """
    def __init__(self,
                 input_services=dict(),
                 adapter_type='simple',
                 inactivity_timeout=10,
                 client_timeout=5):

        self._thread = None
        self._frame = None
        self._history = []
        self._history_tape = []
        self._is_stopped = False

        self._input_adapter = get_adapter(adapter_type)(input_services)
        self._inactivity_timeout = inactivity_timeout
        self._last_access = 0
        self._lock = rwlock.RWLockFair()
        self._event_handler = ClientEventHandler(
                inactivity_timeout=client_timeout)

    def start(self):
        """
        Starts the background thread if it is not running yet. Then
        wait until frames are available.
        """
        # Start threads of dependencies if they're not already started
        if self._input_adapter is not None:
            if self._input_adapter._input_services is not None:
                for _, service in self._input_adapter._input_services.items():
                    service.start()

        if self._thread is None:
            begin = time.time()
            self._last_access = time.time()
            self._thread = threading.Thread(target=self._update)
            self._thread.start()
            self._is_stopped = False
            while self.query()[1] is None:
                time.sleep(0.01)
            log.info('<{:s}> initialized in {:.6f} seconds.'.format(
                    class_name(self), time.time() - begin))

    def stop(self):
        """Marks the thread as stopped so that _update makes a break.
        """
        self._is_stopped = True

    def query(self):
        """Wait till a new frame is available and return the current frame.
        """
        self.start()
        self._last_access = time.time()
        if not self._event_handler.wait(self._inactivity_timeout):
            log.error(
                    '<{:s}>.query timed out (inactive for more than '
                    '{:.2f} seconds.'.format(
                            class_name(self), self._inactivity_timeout))
        self._event_handler.clear()
        with self._lock.gen_rlock():
            return deepcopy(self._history), deepcopy(self._frame)

    def _update(self):
        """Background thread loop
        """
        frames_iterator = self._generator()
        for frame in frames_iterator:
            now = time.time()
            with self._lock.gen_wlock():
                self._history = self._history_tape + [HistoryItem(
                    now, class_name(self))]
                self._frame = frame

            self._event_handler.set()
            self._history_tape = []

            if time.time() - self._last_access > self._inactivity_timeout:
                frames_iterator.close()
                log.warning(
                        'Stopping {} due to inactivity for 10 seconds.'
                        .format(class_name(self)))
                break

            if self._is_stopped:
                break
        self._thread = None

    def _get_inputs(self, input_ids):
        inputs = dict()
        for input_id, (history, frame) in \
                self._input_adapter.get_inputs(input_ids).items():
            self._history_tape.extend(history)
            inputs[input_id] = frame
        return inputs

    @abstractmethod
    def _generator(self):
        """Coroutine generator that yields packages, put in separate thread
        """
        pass
