from queue import SimpleQueue
from typing import Dict, List, Optional

from fluid_ai.base import UiElement

from src.constructor import ModuleConstructor, PipelineConstructor
from src.helper import IPipelineHelper
from src.hybrid.benchmark import Benchmarker
from src.hybrid.logger import Logger
from src.hybrid.worker import Worker
from src.pipeline import PipelineModule


class PipelineManagerHelper:
    """
    Helper of the pipeline manager.

    It manages the UI detection pipeline's worker threads.

    Attributes
    ----------
    key : int
        ID of the instance. It is used to differentiate the instance from other instances
        of this class.
    workers : Dict[PipelineModule, Worker]
        Mapping from pipeline module names to the corresponding pipeline workers.
    textual_elements : List[str]
        List of textual UI class names.
    icon_elements : List[str]
        List of icon UI class names.
    logger : Logger
        Logger to log the UI detection process.
    benchmarker : Optional[Benchmarker]
        Benchmarker to benchmark the UI detection pipeline server. `None` to not
        benchmark the server.
    server_pid : int
        Process ID of the pipeline server.
    """

    key: int
    workers: Dict[PipelineModule, Worker]

    textual_elements: List[str]
    icon_elements: List[str]

    logger: Logger
    benchmarker: Optional[Benchmarker]

    _server_pid: int
    _count: int

    def __init__(
        self,
        key: int,
        constructor: PipelineConstructor,
        logger: Logger,
        benchmarker: Optional[Benchmarker],
        server_pid: int,
    ):
        """
        Parameters
        ----------
        key : int
            ID of the instance. It is used to differentiate the instance from other instances
            of this class.
        constructor : PipelineConstructor
            Constructor of the UI detection pipeline.
        logger : Logger
            Logger to log the UI detection process.
        benchmarker : Optional[Benchmarker]
            Benchmarker to benchmark the UI detection pipeline server. `None` to not
            benchmark the server.
        server_pid : int
            Process ID of the pipeline server.
        """
        self.key = key
        self.workers = {}
        self.logger = logger
        self.benchmarker = benchmarker
        self._server_pid = server_pid

        for name, module in constructor.modules.items():
            self.workers[name] = self._create_worker(name, module)

        self.textual_elements = constructor.textual_elements
        self.icon_elements = constructor.icon_elements

        self._count = 0

    def _create_worker(self, name: PipelineModule, module: ModuleConstructor) -> Worker:
        """Creates a pipeline worker to serve a component of the UI detection pipeline.

        Parameters
        ----------
        name : PipelineModule
            Name of a UI detection pipeline module.
        module : ModuleConstructor
            Constructor of a UI detection pipeline module.

        Returns
        -------
        Worker
            Created pipeline worker.
        """
        return Worker(
            module.func,
            SimpleQueue(),
            module,
            self.logger,
            self._server_pid,
            name.value + str(self.key),
            module.is_thread,
        )

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
            self.workers,
            self.textual_elements,
            self.icon_elements,
            self.logger,
            self.benchmarker,
        )

    def start(self):
        """Starts the pipeline workers"""
        for worker in self.workers.values():
            worker.start()

    def terminate(self, force: bool = False):
        """Terminates the pipeline workers

        It waits until all the workers finish their current jobs.
        """
        for worker in self.workers.values():
            worker.terminate(force)


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
    _workers: Dict[PipelineModule, Worker]
    _channels: Dict[PipelineModule, SimpleQueue]

    def __init__(
        self,
        key: int,
        workers: Dict[PipelineModule, Worker],
        textual_elements: List[str],
        icon_elements: List[str],
        logger: Logger,
        benchmarker: Optional[Benchmarker],
    ):
        """
        Parameters
        ----------
        key : int
            ID of the instance. It is used to differentiate the instance from other instances
            of this class.
        workeres : Dict[PipelineModule, Worker]
            Mapping from pipeline module names to the corresponding pipeline workers.
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
        self._workers = workers
        self.textual_elements = textual_elements
        self.icon_elements = icon_elements
        self._channels = {k: SimpleQueue() for k in workers.keys()}

    def send(self, target: PipelineModule, *args):
        """Sends data to the target UI detection pipeline module

        It does not immediately return the result. Otherwise, `PipelineHelper.wait()`
        must be called to retrieve the result.

        Parameters
        ----------
        target : PipelineModule
            Target UI detection pipeline module.
        *args
            Data to be sent to the module.
        """
        worker = self._workers[target]
        if worker._is_thread:
            worker.channel.put((self._channels[target], args))
        else:
            self._channels[target].put(worker.func(*args, module=worker.module))

    def wait(self, target: PipelineModule) -> List[UiElement]:
        """Waits and gets the result from the target UI detection pipeline module

        This method must be called only after an associated call to the `PipelineHelper.send()`
        method. Otherwise, it will block indefinitely.

        Parameters
        ----------
        target : PipelineModule
            Target UI detection pipeline module.

        Returns
        -------
        List[UiElement]
            Resulting list of UI elements returned from the module.
        """
        return self._channels[target].get()
