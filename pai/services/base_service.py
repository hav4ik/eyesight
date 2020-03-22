import time
import threading
import os

from readerwriterlock import rwlock
from pai.utils import Log


class ClientEventHandler:
    """
    An Event-like class that signals all active clients when a new frame
    is available. Basically holds a pair (Event, timestamp) for each client
    and invokes `wait` and `set` for each of them.
    """
    def __init__(self):
        self.events = dict()

    def wait(self):
        """Invoked from each client's thread to wait for the next frame"""
        ident = threading.get_ident()
        if ident not in self.events:
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait()

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
                if now - event[1] > 5:
                    inactive_clients.append(ident)

        for ident in inactive_clients:
            self.events.pop(ident, None)

    def clear(self):
        """Invoked from each client's thread after a frame is processed"""
        self.events[threading.get_ident()][0].clear()

    def last(self):
        """Time (Unix timestamp) of the last image received by client"""
        return self.events[threading.get_ident()][1]


class BaseStreamService:
    """Base singleton class for each frame streaming service.
    """
    thread = None
    frame = None
    lock = rwlock.RWLockFair()
    last_access = 0
    event_handler = ClientEventHandler()

    def __init__(self):
        """The singleton class is defined by a single thread"""
        self.start_background_thread()

    @classmethod
    def start_background_thread(cls):
        """
        Starts the background streaming thread if it is not running yet. Then
        wait until frames are available.
        """
        if cls.thread is None:
            begin = time.time()
            cls.last_access = time.time()
            cls.thread = threading.Thread(target=cls.update)
            cls.thread.start()
            while cls.get_frame()[1] is None:
                time.sleep(0.01)
            Log.info('{:s} initialized in {:.6f} seconds.'.format(
                    '.'.join([cls.__module__, cls.__name__]),
                    time.time() - begin))

    @classmethod
    def get_frame(cls):
        """Wait till a new frame is available and return the current frame.
        """
        cls.last_access = time.time()
        cls.event_handler.wait()
        cls.event_handler.clear()
        with cls.lock.gen_rlock():
            return cls.event_handler.last(), cls.frame

    @classmethod
    def update(cls):
        """Camera background thread
        """
        frames_iterator = cls.frames()
        for frame in frames_iterator:
            with cls.lock.gen_wlock():
                cls.frame = frame
            cls.event_handler.set()

            if time.time() - cls.last_access > 10:
                frames_iterator.close()
                Log.warning(
                        'Stopping {} due to inactivity for 10 seconds.'
                        .format('.'.join([cls.__module__, cls.__name__])))
                break
        cls.thread = None

    @staticmethod
    def frames():
        """Generator that returns frames from camera
        """
        raise NotImplementedError('Must be implemented by subclasses')
