from queue import Queue
from typing import Dict, List, Optional

from fluid_ai.base import UiElement
from multiprocessing.managers import DictProxy, SyncManager
from threading import Condition

from src.constructor import PipelineConstructor
from src.helper import IPipelineHelper
from src.multiprocess.benchmark import BenchmarkListener, Benchmarker
from src.multiprocess.logger import LogListener, Logger
from src.pipeline import PipelineModule


class PipelineManagerHelper:
    """
    Helper of the pipeline manager.

    It manages the UI detection pipeline's resources.

    Attributes
    ----------
    manager : SyncManager
        Shared resource manager.
    log_listener : LogListener
        Listener of logging events.
    benchmark_listener : Optional[BenchmarkListener]
        Listener of benchmarking events.
    module_channels : Dict[PipelineModule, Queue]
        Mapping from pipeline module names to the corresponding channels for sending
        data to be processed by the modules.
    module_pools : Dict[PipelineModule, DictProxy]
        Mapping from pipeline module names to the corresponding result pools. The
        pipeline workers will post the their outputs with the associated keys on the
        corresponding pools upon finishing their tasks. Consumers can retrieve their
        results using the associated keys.
    textual_elements : List[str]
        List of textual UI class names.
    icon_elements : List[str]
        List of icon UI class names.
    """

    manager: SyncManager
    log_listener: LogListener
    benchmark_listener: Optional[BenchmarkListener]

    module_channels: Dict[PipelineModule, Queue]
    module_pools: Dict[PipelineModule, DictProxy]

    textual_elements: List[str]
    icon_elements: List[str]

    _count: int
    _conditions: DictProxy

    def __init__(
        self,
        pipeline: PipelineConstructor,
        manager: SyncManager,
        log_listener: LogListener,
        benchmark_listener: Optional[BenchmarkListener],
    ):
        """
        Parameters
        ----------
        pipeline : PipelineConstructor
            Constructor of the UI detection pipeline.
        manager : SyncManager
            Shared resource manager.
        log_listener : LogListener
            Listener of logging events.
        benchmark_listener : Optional[BenchmarkListener]
            Listener of benchmarking events.
        """
        self.manager = manager
        self.log_listener = log_listener
        self.benchmark_listener = benchmark_listener

        self.module_channels = {}
        self.module_pools = {}
        for module in pipeline.modules.keys():
            self.module_channels[module] = manager.Queue()
            self.module_pools[module] = manager.dict()

        self.textual_elements = pipeline.textual_elements
        self.icon_elements = pipeline.icon_elements

        self._count = 0
        self._conditions = manager.dict()

    def get_helper(self) -> "PipelineHelper":
        """Instantiates a pipeline helper which can be used by connection handling
        processes to access the UI detection pipeline modules

        Returns
        -------
        PipelineHelper
            Helper for accessing the UI detection pipeline modules.
        """
        key = self._count
        self._count += 1

        condition = self.manager.Condition()
        self._conditions[key] = condition

        return PipelineHelper(
            key,
            self.module_channels,
            self.module_pools,
            self.textual_elements,
            self.icon_elements,
            self.log_listener.get_logger(),
            (
                None
                if self.benchmark_listener is None
                else self.benchmark_listener.get_benchmarker()
            ),
            condition,
            self._conditions,
        )

    def close(self):
        """Closes the resources used by the UI detection pipeline"""
        self.manager.shutdown()


class PipelineHelper(IPipelineHelper):
    """
    Helper for accessing the UI detection pipeline modules.

    Attributes
    ----------
    key : int
        ID of the instance. It is used to differentiate the instance from other instances
        of this class.
    logger : ILogger
        Logger to log the UI detection process.
    benchmarker : Optional[IBenchmarker]
        Benchmarker to benchmark the UI detection pipeline server. `None` to not
        benchmark the server.
    textual_elements : List[str]
        List of textual UI class names.
    icon_elements : List[str]
        List of icon UI class names.
    module_channels : Dict[PipelineModule, Queue]
        Mapping from pipeline module names to the corresponding channels for sending
        data to be processed by the modules.
    module_pools : Dict[PipelineModule, DictProxy]
        Mapping from pipeline module names to the corresponding result pools. The
        pipeline workers will post the their outputs with the associated keys on the
        corresponding pools upon finishing their tasks. Consumers can retrieve their
        results using the associated keys.
    """

    key: int

    module_channels: Dict[PipelineModule, Queue]
    module_pools: Dict[PipelineModule, DictProxy]

    _condition: Condition
    _manager_conditions: DictProxy

    def __init__(
        self,
        key: int,
        module_channels: Dict[PipelineModule, Queue],
        module_pools: Dict[PipelineModule, DictProxy],
        textual_elements: List[str],
        icon_elements: List[str],
        logger: Logger,
        benchmarker: Optional[Benchmarker],
        condition: Condition,
        manager_conditions: DictProxy,
    ):
        """
        Parameters
        ----------
        key : int
            ID of the instance. It is used to differentiate the instance from other instances
            of this class.
        module_channels : Dict[PipelineModule, Queue]
            Mapping from pipeline module names to the corresponding channels for sending
            data to be processed by the modules.
        module_pools : Dict[PipelineModule, DictProxy]
            Mapping from pipeline module names to the corresponding result pools. The
            pipeline workers will post the their outputs with the associated keys on the
            corresponding pools upon finishing their tasks. Consumers can retrieve their
            results using the associated keys.
        textual_elements : List[str]
            List of textual UI class names.
        icon_elements : List[str]
            List of icon UI class names.
        logger : ILogger
            Logger to log the UI detection process.
        benchmarker : Optional[IBenchmarker]
            Benchmarker to benchmark the UI detection pipeline server. `None` to not
            benchmark the server.
        condition : Condition
            Condition variable, used to be notified when the results are available.
        manager_conditions : DictProxy
            Condition variable pool, used to clean up the condition variable used by
            this helper.
        """
        super().__init__(logger, benchmarker, textual_elements, icon_elements)
        self.key = key
        self.module_channels = module_channels
        self.module_pools = module_pools
        self._condition = condition
        self._manager_conditions = manager_conditions

    def send(self, target: PipelineModule, *args):
        self.module_channels[target].put((self.key, self._condition, args))

    def wait(self, target: PipelineModule) -> List[UiElement]:
        with self._condition:
            self._condition.wait_for(
                lambda: self.module_pools[target].get(self.key, None) is not None
            )
            return self.module_pools[target].pop(self.key)

    def __del__(self):
        self._manager_conditions.pop(self.key, None)
