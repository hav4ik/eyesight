from ABC import abstractmethod
import threading
from collections import deque
from enum import Enum

from .base_service import BaseService
from .adapters import get as get_adapter
from .adapters import MultiAdapter


class BiRunnerService(BaseService):

    RunnerType = Enum('RunnerType', 'front back')

    def __init__(self,
                 slow_inputs=dict(),
                 fast_inputs=dict(),
                 slow_adapter_type=get_adapter('simple'),
                 fast_adapter_type=get_adapter('simple'),
                 *args, **kwargs):

        adapter = MultiAdapter(
                slow_adapter_type(slow_inputs),
                fast_adapter_type(fast_inputs))

        # The fast input cache will be stored in deques, since
        # we want to save for multiple checkpoints (for the
        # slow runner to catch up)
        self._fast_input_names = fast_inputs.keys()
        self._fast_input_cache = dict(
                (name, deque()) for name in self._fast_input_names)
        self._fast_input_received = threading.Event()

        # The slow input cache, on other hand, is needed to be
        # read only once (by the slow runner)
        self._slow_input_names = slow_inputs.keys()
        self._slow_input_cache = dict(
                (name, None) for name in self._slow_input_names)
        self._slow_input_received = threading.Event()

        # Thread type (front runner / catch-up runner) switcher
        self._front_runner_thread = None
        self._front_runner_id = None
        self._back_runner_thread = None
        self._back_runner_id = None

        super().__init__(adapter=adapter, *args, **kwargs)

    def _get_runner_type(self):
        if self._front_runner_id is None or self._back_runner_id is None:
            raise ValueError(
                    'Front and Back runners are not initialized. Have you '
                    'started the service using ServiceClass.start()?')

        thread_id = threading.get_ident()
        if self._front_runner_id == thread_id:
            return BiRunnerService.RunnerType.front
        elif self._back_runner_id == thread_id:
            return BiRunnerService.RunnerType.back
        else:
            raise RuntimeError(
                    "This method was called from neither front or back "
                    "runner's threads. Please make sure that this method is "
                    "not called from outside of the BiRunnerService class.")

    def _switch_runners(self):
        runner_type = self._get_runner_type()
        if runner_type != BiRunnerService.RunnerType.back:
            raise RuntimeError(
                    "BiRunnerService._switch_runners should only be called "
                    "from the back runner's thread. Please make sure that "
                    "this method is not called from outside of the "
                    "BiRunnerService class.")

        self._front_runner_id, self._back_runner_id = \
            self._back_runner_id, self._front_runner_id
        self._front_runner_thread, self._back_runner_thread = \
            self._back_runner_thread, self._front_runner_thread

    def _slow_listener(self):
        while not self._is_stopped:
            self._slow_input_cache = dict(zip(
                self._slow_input_names,
                self._get_inputs(self._slow_input_names)))
            self._slow_input_received.set()

    def _fast_listener(self):
        while not self._is_stopped:
            rets = self._get_inputs(self._fast_input_names)
            for name, ret in zip(self._fast_input_names, rets):
                self._fast_input_cache[name].append(ret)
            self._fast_input_received.set()

    def _runner_thread(self, index):
        while not self._is_stopped:
            self._slow_input_received.wait()
            if not self._get_runner_type() == BiRunnerService.RunnerType.back:
                raise RuntimeError(
                        'Only back runner can wait for slow inputs.')

            output_iterator = self._runner(self.slow_input_cache)
            self._slow_input_received.clear()

            for output in output_iterator:
                if self._is_stopped:
                    break

    @abstractmethod
    def _runner(self, slow_inputs):
        pass

    def _generator(self):
        while True:
            pass
