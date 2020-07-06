from abc import ABCMeta, abstractmethod
from collections import deque
import threading
import time
from ..utils.generic_utils import log


# Empty history and None value
empty_query = ([], None)


class BaseInputAdapter(metaclass=ABCMeta):
    """Base class for input adapters.

    # Arguments:
        input_services: a `dict` {'service_name': service}, where service
            inherits from BaseService
    """
    def __init__(self, input_services=dict()):
        if isinstance(input_services, list):
            self._input_services = dict(enumerate(input_services))
        elif isinstance(input_services, dict):
            self._input_services = input_services
        else:
            raise ValueError(
                    '`input_services` should either be a dict or a list')

    def get_inputs(self, *input_ids):
        """
        Given input ids, return a tuple of service outputs in same order.
        This wrapper around the internal function `_get_inputs_internal`
        also ensures that only existing input_ids are passed to the derived
        classes' implementation.
        """
        if len(input_ids) == 0:
            return empty_query

        existing_ids = [
                sid for sid in input_ids if sid in self._input_services]

        if len(existing_ids) == 0:
            # Empty history and empty value
            return tuple(empty_query for _ in input_ids)

        output_dict = self._get_inputs_internal(*existing_ids)
        ret = tuple(
                output_dict[sid] if sid in output_dict else empty_query
                for sid in input_ids)

        return ret[0] if len(ret) == 1 else ret

    def start(self):
        pass

    def stop(self, wait=False):
        pass

    @abstractmethod
    def _get_inputs_internal(self, *input_ids):
        """Retrieve a list of outputs and returns a dict {input_id: value}
        """
        return NotImplemented


class SimpleAdapter(BaseInputAdapter):
    """Simply get what it is asked

    # Arguments:
        input_services: a `dict` {'service_name': service}, where service
            inherits from BaseService
    """
    def __init__(self, input_services):
        super().__init__(input_services)

    def _get_inputs_internal(self, *input_ids):
        return dict((input_id, self._input_services[input_id].query())
                    for input_id in input_ids)


class LatestAdapter(BaseInputAdapter):
    """
    Stores the latest frame of each service and returns when updates
    are available

    IMPORTANT: no locks are needed since adapters are just passing objects
        through itself without modifying them. So, every assignment here
        are just name assignments and thus, thread safe.

    # Arguments:
        input_services: a `dict` {'service_name': service}, where service
            inherits from BaseService
    """
    def __init__(self, input_services):
        super().__init__(input_services)
        self._saved_outputs = dict(
                (input_id, None)
                for input_id in input_services.keys())
        self._threads = dict(
                (input_id, None)
                for input_id in input_services.keys())
        self._updated_event = threading.Event()
        self._stopped = True

    def start(self):
        self._stopped = False
        for input_id in self._input_services:
            if self._threads[input_id] is None:
                self._threads[input_id] = threading.Thread(
                        target=self._update_cache, args=(input_id, ))
                self._threads[input_id].start()

    def stop(self, wait=False):
        self._stopped = True
        if wait and len(self._threads) > 0:
            for _, thread in self._threads.items():
                if thread is not None:
                    thread.join()

    def _update_cache(self, input_id):
        while True:
            if self._stopped:
                break
            self._saved_outputs[input_id] = \
                self._input_services[input_id].query()
            self._updated_event.set()

        self._threads[input_id] = None

    def _get_inputs_internal(self, *input_ids):
        if self._stopped:
            log.error('The adapter is stopped, cannot retrieve anything')
            return dict((input_id, empty_query) for input_id in input_ids)

        self._updated_event.wait()
        self._updated_event.clear()
        ret = dict((input_id, empty_query) for input_id in input_ids)
        for input_id in input_ids:
            val = self._saved_outputs[input_id]
            if val is not None:
                ret[input_id] = val
        return ret


class CachedAdapter(BaseInputAdapter):
    """
    Stores the cached latest frame of each service and returns it when
    requested (so the latest frame can be returned multiple times, if
    the input frequency is lower than query frequency).

    IMPORTANT: no locks are needed since adapters are just passing objects
        through itself without modifying them. So, every assignment here
        are just name assignments and thus, thread safe.

    # Arguments:
        input_services: a `dict` {'service_name': service}, where service
            inherits from BaseService
    """
    def __init__(self, input_services):
        super().__init__(input_services)
        self._saved_outputs = dict(
                (input_id, None)
                for input_id in input_services.keys())
        self._threads = dict(
                (input_id, None)
                for input_id in input_services.keys())
        self._stopped = True

    def start(self):
        self._stopped = False
        for input_id in self._input_services:
            if self._threads[input_id] is None:
                self._threads[input_id] = threading.Thread(
                        target=self._update_cache, args=(input_id, ))
                self._threads[input_id].start()

    def stop(self, wait=False):
        self._stopped = True
        if wait and len(self._threads) > 0:
            for _, thread in self._threads.items():
                if thread is not None:
                    thread.join()

    def _update_cache(self, input_id):
        while True:
            if self._stopped:
                break
            self._saved_outputs[input_id] = \
                self._input_services[input_id].query()

        self._threads[input_id] = None

    def _get_inputs_internal(self, *input_ids):
        if self._stopped:
            log.error('The adapter is stopped, cannot retrieve anything')
            return dict((input_id, empty_query) for input_id in input_ids)

        ret = dict((input_id, empty_query) for input_id in input_ids)
        for input_id in input_ids:
            val = self._saved_outputs[input_id]
            if val is not None:
                ret[input_id] = val
        return ret


class SyncAdapter(BaseInputAdapter):
    """
    Returns only the frames with closest timestamps, as illustrated below:
    (by "." we denote frames, ":" - the ones in cache, and "@" denotes
    returned frames)

                            Returned frames
                                   v
      service 1 |     .      .     @
      service 2 |..................@::::::
      service 3 | .  .  .  .  .  .  @  :
                 -------------------------> Timeline (by source stamp)

    IMPORTANT: no locks are used here not because the operations are thread
        safe, but because we can tolerate thread safety issues! The append
        and pop for `collections.deque` are thread-safe on both sides, and
        the only `pop` operation happens on the left side in the same thread
        as other reading operations (in `_get_inputs_internal`), so no
        invalid element access is possible. The only `append` operation
        happens only on the right side and only in `_update_inputs` threads.
        By the scheme above, we can tolerate yielding a bit earlier frames,
        since everything will still be in cache.

    # Arguments:
        input_services: a `dict` {'service_name': service}, where service
            inherits from BaseService
        max_cache_size: number of frames to store in cache. If set to None,
            the cache is unlimited (beware memory issues!)
        adjust_slowest: only yield new values when the slowest service
            (which also means the one with most delay) yields.
        fps_cache: the number of last frames to take into account while
            calculating service's performance (if `adjust_slowest` is True).
    """
    def __init__(self,
                 input_services,
                 max_cache_size=200,
                 adjust_slowest=True,
                 fps_cache=50):

        super().__init__(input_services)
        self._cache = dict(
                (service_name, deque(maxlen=max_cache_size))
                for service_name in input_services.keys())

        self._threads = dict(
                (input_id, None)
                for input_id in input_services)

        self._adjust_slowest = adjust_slowest
        if adjust_slowest:
            assert fps_cache > 1
            self._fps_cache = fps_cache
            self._fps = dict((input_id, 0) for input_id in input_services)

        self._received_event = threading.Event()
        self._stopped = True

    def start(self):
        self._stopped = False
        for input_id in self._input_services:
            if self._threads[input_id] is None:
                if self._adjust_slowest:
                    self._fps[input_id] = 0
                self._threads[input_id] = threading.Thread(
                        target=self._update_inputs, args=(input_id, ))
                self._threads[input_id].start()

    def stop(self, wait=False):
        self._stopped = True
        if wait and len(self._threads) > 0:
            for _, thread in self._threads.items():
                if thread is not None:
                    thread.join()

    def _update_inputs(self, input_id):
        if self._adjust_slowest:
            start_time = time.time()
            backlog = deque(maxlen=self._fps_cache)

        while True:
            if self._stopped:
                break

            # Make query to the input service
            last = self._input_services[input_id].query()

            # Re-calibrating FPS
            if self._adjust_slowest:
                backlog.append(time.time())
                if len(backlog) == backlog.maxlen:
                    self._fps[input_id] = len(backlog) / (
                            backlog[-1] - backlog[0])
                else:
                    self._fps[input_id] = len(backlog) / (
                            backlog[-1] - start_time)

            # Put the value to cache
            cache = self._cache[input_id]
            cache.append(last)
            if len(cache) == cache.maxlen:
                log.warning('Cache for `{}` reached max capacity of {}'
                            .format(input_id, cache.maxlen))

            # Set the event only if this is the slowest service
            if self._adjust_slowest:
                min_fps = min(self._fps.values())
                if not self._fps[input_id] == min_fps:
                    continue
            self._received_event.set()

        self._threads[input_id] = None

    def _get_inputs_internal(self, *input_ids):
        if self._stopped:
            log.error('The adapter is stopped, cannot retrieve anything')
            return dict((input_id, empty_query) for input_id in input_ids)

        self._received_event.wait()
        self._received_event.clear()
        ret = dict()

        earliest = None
        for input_id in input_ids:
            cache = self._cache[input_id]
            if len(cache) == 0:
                continue

            # Index explanation:
            # deque -> last element -> history -> first stamp
            t = cache[-1][0][0].timestamp
            if earliest is None:
                earliest = t
            else:
                earliest = min(t, earliest)

        for input_id in input_ids:
            cache = self._cache[input_id]
            if len(cache) == 0:
                # Return empty history and None as value
                ret[input_id] = empty_query
                continue

            ret[input_id] = cache[-1]

            # Index explanation:
            # deque -> first element -> history -> first stamp
            while len(cache) > 1 and \
                    cache[0][0][0].timestamp < earliest:
                ret[input_id] = cache.popleft()

            # Choose the one that is closer
            d0 = abs(ret[input_id][0][0].timestamp - earliest)
            d1 = abs(cache[0][0][0].timestamp - earliest)
            if d1 < d0:
                if len(cache) > 1:
                    ret[input_id] = cache.popleft()
                else:
                    ret[input_id] = cache[0]

        return ret


class MultiAdapter(BaseInputAdapter):
    """
    Combination of adapters, so that the user can flexibly define
    data reading patterns.
    """
    def __init__(self, input_adapters):
        services = dict()
        for adapter in input_adapters:
            services.extend(adapter._input_services)
        self._adapters = input_adapters
        super().__init__(services)

    def _get_inputs_internal(self, *input_ids):
        ret_dict = dict()
        for adapter in self._input_adapters:
            sub_input_ids = list(set(input_ids).union(
                set(adapter._input_services.keys())))
            ret_dict.extend(adapter.get_input(sub_input_ids))
        return sub_input_ids


def get(identifier):
    """An easy interface for one to load any adapters available.
    """
    if isinstance(identifier, str):
        if identifier == 'simple':
            return SimpleAdapter
        elif identifier == 'latest':
            return LatestAdapter
        elif identifier == 'sync':
            return SyncAdapter
        elif identifier == 'cached':
            return CachedAdapter
    elif isinstance(identifier, BaseInputAdapter):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'adapter identifier:', identifier)
