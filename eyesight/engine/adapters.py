from abc import ABCMeta, abstractmethod
from copy import deepcopy
import asyncio


class BaseInputAdapter(metaclass=ABCMeta):
    """Base class for input adapters.
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
    """
    def __init__(self, input_services):
        super().__init__(input_services)
        self._saved_outputs = dict((k, None) for k in input_services)
        self._coroutines = [self.get(input_id)
                            for input_id in input_services]
        self._lock = asyncio.Lock()

    @asyncio.coroutine
    def get(self, input_id):
        while True:
            with self._lock:
                self._saved_outputs[input_id] = \
                    self._input_services[input_id].query().copy()
            yield self._saved_outputs[input_id]

    async def get_inputs(self, *input_ids):
        await asyncio.wait(self._coroutines,
                           return_when=asyncio.FIRST_COMPLETED)
        with self._lock:
            # TODO: too many copies!
            return dict((input_id, deepcopy(self._saved_outputs))
                        for input_id in input_ids)


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
