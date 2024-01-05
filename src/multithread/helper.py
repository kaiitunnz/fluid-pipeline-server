from queue import SimpleQueue
from typing import Dict, List, Optional

from fluid_ai.base import UiElement

from src.benchmark import Benchmarker
from src.constructor import ModuleConstructor, PipelineConstructor
from src.multithread.loadbalancer import LoadBalancer
from src.multithread.logger import Logger
from src.multithread.worker import Worker
from src.pipeline import PipelineModule, IPipelineHelper


class PipelineManagerHelper:
    """
    Helper of the pipeline manager.

    It manages the UI detection pipeline's worker threads.

    Attributes
    ----------
    channels: Dict[PipelineModule, LoadBalancer]
        Mapping from pipeline module names to the corresponding load balancers.
    textual_elements : List[str]
        List of textual UI class names.
    icon_elements : List[str]
        List of icon UI class names.
    logger : Logger
        Logger to log the UI detection process.
    benchmarker : Optional[Benchmarker]
        Benchmarker to benchmark the UI detection pipeline server. `None` to not
        benchmark the server.
    num_instances : int
        Number of instances of the UI detection pipeline.
    """

    channels: Dict[PipelineModule, LoadBalancer]

    textual_elements: List[str]
    icon_elements: List[str]

    logger: Logger
    benchmarker: Optional[Benchmarker]

    num_instances: int
    _count: int

    def __init__(
        self,
        constructor: PipelineConstructor,
        logger: Logger,
        benchmarker: Optional[Benchmarker],
        num_instances: int,
    ):
        """
        Parameters
        ----------
        constructor : PipelineConstructor
            Constructor of the UI detection pipeline.
        logger : Logger
            Logger to log the UI detection process.
        benchmarker : Optional[Benchmarker]
            Benchmarker to benchmark the UI detection pipeline server. `None` to not
            benchmark the server.
        num_instances : int
            Number of instances of the UI detection pipeline.
        """
        self.channels = {}
        self.num_instances = num_instances
        self.logger = logger
        self.benchmarker = benchmarker

        for name, module in constructor.modules.items():
            self.channels[name] = self._create_module_instances(name, module)

        self.textual_elements = constructor.textual_elements
        self.icon_elements = constructor.icon_elements

        self._count = 0

    def _create_module_instances(
        self, name: PipelineModule, module: ModuleConstructor
    ) -> LoadBalancer:
        """Creates a load balancer that manages multiple instances of the given pipeline
        module

        Parameters
        ----------
        name : PipelineModule
            Name of a UI detection pipeline module.
        module : ModuleConstructor
            Constructor of a UI detection pipeline module.

        Returns
        -------
        LoadBalancer
            Created load balancer.
        """
        workers = [
            Worker(
                module.func, SimpleQueue(), module(), self.logger, name.value + str(i)
            )
            for i in range(self.num_instances)
        ]
        return LoadBalancer(workers, self.logger)

    def get_helper(self) -> "PipelineHelper":
        """Instantiates a pipeline helper which can be used by connection handling
        threads to access the UI detection pipeline modules

        Returns
        -------
        PipelineHelper
            Helper for accessing the UI detection pipeline modules.
        """
        key = self._count
        self._count += 1
        return PipelineHelper(
            key,
            self.channels,
            self.textual_elements,
            self.icon_elements,
            self.logger,
            self.benchmarker,
        )

    def get_warmup_helper(self, i: int) -> "WarmupHelper":
        return WarmupHelper(
            i,
            self.channels,
            self.textual_elements,
            self.icon_elements,
            self.logger,
            self.benchmarker,
        )

    def start(self):
        """Starts the pipeline workers"""
        for balancer in self.channels.values():
            balancer.start()

    def terminate(self, force: bool = False):
        """Terminates the pipeline workers"""
        for balancer in self.channels.values():
            balancer.terminate(force)


class PipelineHelper(IPipelineHelper):
    """
    Helper for accessing the UI detection pipeline modules.

    Attributes
    ----------
    key : int
        ID of the instance. It is used to differentiate the instance from other instances
        of this class.
    textual_elements : List[str]
        List of textual UI class names.
    icon_elements : List[str]
        List of icon UI class names.
    logger : Logger
        Logger to log the UI detection process.
    benchmarker : Optional[Benchmarker]
        Benchmarker to benchmark the UI detection pipeline server. `None` to not
        benchmark the server.
    """

    key: int

    _channels: Dict[PipelineModule, LoadBalancer]
    _res_channels: Dict[PipelineModule, SimpleQueue]

    def __init__(
        self,
        key: int,
        channels: Dict[PipelineModule, LoadBalancer],
        textual_elements: List[str],
        icon_elements: List[str],
        logger: Logger,
        benchmarker: Optional[Benchmarker],
    ):
        """
        Parameters
        ----------
        key : int
            ID of the instance. It is used to differentiate the instance from other
            instances of this class.
        channels : Dict[PipelineModule, LoadBalancer]
            Mapping from pipeline module names to the corresponding load balancers.
        textual_elements : List[str]
            List of textual UI class names.
        icon_elements : List[str]
            List of icon UI class names.
        logger : Logger
            Logger to log the UI detection process.
        benchmarker : Optional[Benchmarker]
            Benchmarker to benchmark the UI detection pipeline server. `None` to not
            benchmark the server.
        """
        super().__init__(logger, benchmarker, textual_elements, icon_elements)
        self.key = key
        self._channels = channels
        self._res_channels = {module: SimpleQueue() for module in channels.keys()}

    def send(self, target: PipelineModule, *args):
        self._channels[target].send((self._res_channels[target], args))

    def wait(self, target: PipelineModule) -> List[UiElement]:
        return self._res_channels[target].get()

    def get_num_instances(self) -> int:
        """Gets the number of instances of the UI detection pipeline

        Returns
        -------
        int
            Number of instances of the UI detection pipeline.
        """
        if len(self._channels) == 0:
            return 0
        return len(tuple(self._channels.values())[0].workers)


class WarmupHelper(PipelineHelper):
    def send(self, target: PipelineModule, *args):
        self._channels[target].sendi(self.key, (self._res_channels[target], args))
