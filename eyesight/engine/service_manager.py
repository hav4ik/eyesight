from abc import ABCMeta
import time

from .base_service import BaseService
from ..utils.generic_utils import log
from ..utils.generic_utils import class_name


class ServiceManager(metaclass=ABCMeta):
    """Manage the services in a centralized way.
    """
    def __init__(self, *services):
        for service in services:
            self.add_recursive(service)

    def add(self, service, name=None):
        """
        Adds a new service to the service manager, under certain name. The
        service is then accessible with `self.name`.

        # Arguments:
            service: an instance of BaseService or inherited class from it.
            name: a key to associate the service with. If None, the name is
                chosen automatically.

        # Returns: the service itself

        # Raises:
            ValueError: if this manager already has a service with this name
        """
        if name is None:
            base_name = type(service).__name__
            i = 1
            ith_name = base_name + str(i)
            while hasattr(self, ith_name):
                i += 1
                ith_name = base_name + str(i)
            name = ith_name

        if hasattr(self, name):
            raise ValueError('Attribute "{}" already exists.'.format(name))
        setattr(self, name, service)
        self._register(name)
        return service

    def add_recursive(self, service):
        """
        Adds the specified service to the manager, and its whole dependency
        tree as well.

        # Arguments:
            service: an instance of BaseService or inherited class from it.

        # Returns: the service itself.
        """
        self.add(service)
        for _, child_service in service._input_adapter._input_services.items():
            self.add_recursive(child_service)
        return service

    def start(self):
        """Starts all services that are assigned to the manager at once.
        """
        services = self.enlist_services()
        for service in services:
            service.start()

    def stop(self, warning_interval=10, rage_resign_threshold=-1):
        """
        Sends stop signals to each thread. Beware that it is up to each
        service's implementation to ensure that it will be able to stop
        correctly.

        # Arguments:
            warning_interval: Integer. Display warning messages every warning
                interval if some services are not stopped yet.
            rage_resign_threshold: Integer. Stop waiting for threads to stop
                after this interval. Might be useful if you require a high
                level of responsiveness. If -1, never give up on waiting.

        # Returns:
            True if all services are successfully stopped. False otherwise.
        """
        services = self.enlist_services()
        begin = time.time()
        warning_threshold = warning_interval

        for service in services:
            service.stop()

        while True:
            all_stopped = True
            not_stopped = []
            for service in services:
                if service._thread is not None:
                    all_stopped = False
                    not_stopped.append(class_name(service))

            if not all_stopped:
                diff = time.time() - begin
                if diff > warning_threshold:
                    log.warning(
                            'The following services are not responding after '
                            '{:d} seconds: {}'.format(
                                warning_interval, ', '.join(not_stopped)))

                if diff > rage_resign_threshold and rage_resign_threshold >= 0:
                    log.warning('Resign on wairning for services to stop.')
                    return False
                time.sleep(0.01)
            else:
                break
        return True

    def enlist_services(self):
        """Returns a list of attributes that are instances of `BaseService`.
        """
        services = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, BaseService):
                services.append(attr)
        return services

    def _register(self, name):
        """Sets the manager of a service with given name to self.

        # Arguments:
            name: the key assigned to the service for current manager.

        # Raises:
            KeyError: if a service with given key was not found.
            ValueError: if the attribute under the key is not a BaseService
        """
        if not hasattr(self, name):
            raise KeyError('No service with name {}'.format(name))
        if not isinstance(getattr(self, name), BaseService):
            raise ValueError('The attribute "{}" is not a service'
                             .format(name))
        service = getattr(self, name)
        service.manager = self
