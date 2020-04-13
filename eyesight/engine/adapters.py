from abc import ABCMeta, abstractmethod
from copy import deepcopy
from collections import deque
import asyncio
import time


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

    @abstractmethod
    def get_inputs(self, input_ids):
        """Retrieve a list of outputs
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

    def get_inputs(self, *input_ids):
        return dict((input_id, self._input_services[input_id].query())
                    for input_id in input_ids)


class LatestAdapter(BaseInputAdapter):
    """
    Stores the latest frame of each service and returns when updates
    are available

    # Arguments:
        input_services: a `dict` {'service_name': service}, where service
            inherits from BaseService
    """
    def __init__(self, input_services):
        super().__init__(input_services)
        self._saved_outputs = dict((k, None) for k in input_services.keys())
        self._coroutines = [
                self.get(input_id) for input_id in input_services.keys()]
        self._coroutine_ids = dict(
                zip(input_services.keys(), range(len(input_services))))
        self._lock = asyncio.Lock()

    @asyncio.coroutine
    def get(self, input_id):
        while True:
            with self._lock:
                self._saved_outputs[input_id] = \
                    self._input_services[input_id].query()
            yield self._saved_outputs[input_id]

    async def get_inputs(self, *input_ids):
        await asyncio.wait(self._coroutines[self._coroutine_ids[input_ids]],
                           return_when=asyncio.FIRST_COMPLETED)
        with self._lock:
            # TODO: too many copies!
            return dict((input_id, deepcopy(self._saved_outputs))
                        for input_id in input_ids)


class SyncAdapter(BaseInputAdapter):
    """Returns only the frames with closest timestamps.

    # Arguments:
        input_services: a `dict` {'service_name': service}, where service
            inherits from BaseService
        max_cache: number of frames to store in cache. If set to None,
            the cache is unlimited (beware memory issues!)
    """
    def __init__(self, input_services, max_cache_size=200):
        super().__init__(input_services)
        self._cache = dict()
        for service_name, service in input_services.items():
            self._cache[service_name] = deque(maxlen=max_cache_size)
        self._coroutines = [
                self.get(input_id) for input_id in input_services]
        self._coroutine_ids = dict(
                zip(input_services.keys(), range(len(input_services))))
        self._lock = asyncio.Lock()

    @asyncio.coroutine
    def get(self, input_id):
        while True:
            with self._lock:
                last = self._input_services[input_id].query()
                self._cache[input_id].append(last)
            yield last

    async def get_inputs(self, *input_ids):
        await asyncio.wait(self._coroutines[self._coroutine_ids[input_ids]],
                           return_when=asyncio.FIRST_COMPLETED)
        ret = dict()
        with self._lock:
            earliest = time.time()
            for input_id in input_ids:
                # deque -> last element -> history -> last entry
                t = self._cache[input_id][-1][0][-1].timestamp
                earliest = min(t, earliest)

            for input_id in input_ids:
                # deque -> first element -> history -> last entry
                while self._cache[input_id][0][0][-1].timestamp < earliest:
                    ret[input_id] = self._cache[input_id].popleft()

        # We can do that outside the lock because the `ret` contains only
        # elements that got popped out already.
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

    def get_inputs(self, *input_ids):
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
    elif isinstance(identifier, BaseInputAdapter):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'adapter identifier:', identifier)
