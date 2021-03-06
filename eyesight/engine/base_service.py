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

    def register(self):
        """Invoked from each client's thread to register ahead of time"""
        ident = threading.get_ident()
        if ident not in self.events:
            self.events[ident] = [threading.Event(), time.time()]

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
        adapter: an instance of `adapters.BaseInputAdapter`.
        inactivity_timeout: (in seconds) if inactive more than this amount of
            time, will automatically shut down.
        client_timeout: (in seconds) disconnect client services if they were
            inactive for this amount of time.
        lock_free: (experimental) whether we should follow the lock-free
            paradigm and rely minimally on copies. This would require an
            additional level of carefulness while handling and writing the
            services.
        no_copy: (experimental) return values without deep-copying them. This
            will require certain level of carefulness when handling results of
            `query()`.
    """
    def __init__(self,
                 adapter=get_adapter('simple')(dict()),
                 inactivity_timeout=10,
                 client_timeout=5,
                 lock_free=False,
                 no_copy=True):

        self._thread = None
        self._frame = None
        self._history = []
        self._history_tape = []
        self._is_stopped = False
        self._manager = None

        self._input_adapter = adapter
        self._inactivity_timeout = inactivity_timeout
        self._last_access = 0
        self._lock_free = lock_free
        self._no_copy = no_copy

        if not lock_free:
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

        # Start the adapter
        self._input_adapter.start()

        # Start the service thread
        if self._thread is None:
            begin = time.time()
            self._last_access = time.time()
            self._thread = threading.Thread(target=self._update)
            self._thread.start()
            self._is_stopped = False
            while self.query()[1] is None:
                time.sleep(0.01)
            log.debug('<{:s}> initialized in {:.6f} seconds.'.format(
                    class_name(self), time.time() - begin))

    def stop(self, wait=False):
        """Marks the thread as stopped so that _update makes a break.
        """
        self._is_stopped = True
        if wait and self._thread is not None:
            self._thread.join()
        self._input_adapter.stop(wait)

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
        if self._lock_free:
            if self._no_copy:
                return self._history, self._frame
            else:
                return deepcopy(self._history), deepcopy(self._frame)
        else:
            with self._lock.gen_rlock():
                if self._no_copy:
                    return self._history, self._frame
                else:
                    return deepcopy(self._history), deepcopy(self._frame)

    def _update(self):
        """Background thread loop
        """
        frames_iterator = self._generator()
        for frame in frames_iterator:
            now = time.time()

            if self._lock_free:
                self._history = self._history_tape + [HistoryItem(
                    now, class_name(self))]
                self._frame = frame
            else:
                with self._lock.gen_wlock():
                    self._history = self._history_tape + [HistoryItem(
                        now, class_name(self))]
                    self._frame = frame

            self._event_handler.set()
            self._history_tape = []

            if time.time() - self._last_access > self._inactivity_timeout:
                frames_iterator.close()
                log.warning(
                        'Stopping <{:s}> due to inactivity for 10 seconds.'
                        .format(class_name(self)))
                break

            if self._is_stopped:
                log.debug(
                        '<{:s}> received stop signal.'
                        .format(class_name(self)))
                break
        self._thread = None

    def _get_inputs(self, *input_ids):
        """Retrieves output value from the adapters
        """
        if len(input_ids) == 0:
            return None

        adapter_ret = self._input_adapter.get_inputs(*input_ids)
        if len(input_ids) == 1:
            adapter_ret = tuple((adapter_ret, ))
        inputs = [frame for history, frame in adapter_ret]

        # Need to deal with the history
        for history, frame in adapter_ret:

            # Always make sure that the history is sorted by timestamps
            # and don't contain duplicates
            self._history_tape.sort(key=lambda x: x.timestamp)
            count_common = len(
                    [x for x in zip(self._history_tape, history)
                     if (x[0].timestamp == x[1].timestamp and
                         x[0].invoker == x[1].invoker)])

            self._history_tape.extend(history[count_common:])

        return tuple(inputs) if len(inputs) > 1 else inputs[0]

    def _safe_resolve_input(self, value, readonly=True):
        if not readonly:
            if self._no_copy:
                return deepcopy(value)
            else:
                return value

    @property
    def manager(self):
        return self._manager

    @manager.setter
    def manager(self, value):
        self._manager = value

    @abstractmethod
    def _generator(self):
        """Coroutine generator that yields packages, put in separate thread
        """
        NotImplemented
