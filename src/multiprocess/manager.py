import logging
from typing import Dict, Optional

from multiprocessing.managers import SyncManager

import src.benchmark as bench
from src.constructor import ModuleConstructor, PipelineConstructor
from src.multiprocess.benchmark import BenchmarkListener
from src.multiprocess.helper import PipelineHelper, PipelineManagerHelper
from src.multiprocess.logger import LogListener, Logger
from src.multiprocess.worker import Worker
from src.pipeline import PipelineModule


class PipelineManager:
    """
    Manager of the UI detection pipeline.

    It manages all the resources used by the UI detection pipeline, especially the
    worker processes.

    Attributes
    ----------
    pipeline : PipelineConstructor
        Constructor of the UI detection pipeline.
    workers : Dict[PipelineModule, Worker]
        Mapping from pipeline module names to the corresponding pipeline workers.
    log_listener: LogListener
        Listener of logging events.
    logger : Logger
        Logger to log the UI detection process.
    benchmark_listener : Optional[BenchmarkListener]
        Listener of benchmarking events.
    """

    pipeline: PipelineConstructor
    workers: Dict[PipelineModule, Worker]

    _helper: PipelineManagerHelper

    log_listener: LogListener
    logger: logging.Logger

    benchmark_listener: Optional[BenchmarkListener]

    def __init__(
        self,
        pipeline: PipelineConstructor,
        manager: SyncManager,
        logger: logging.Logger,
        benchmarker: Optional[bench.Benchmarker],
    ):
        """
        Parameters
        ----------
        pipeline : PipelineConstructor
            Constructor of the UI detection pipeline.
        manager : SyncManager
            Shared resource manager.
        logger : Logger
            Logger to log the UI detection process.
        benchmark_listener : Optional[BenchmarkListener]
            Benchmarker to benchmark the UI detection pipeline server. `None` to not
            benchmark the server.
        """
        self.pipeline = pipeline
        self.workers = {}
        self.log_listener = LogListener(logger, manager.Queue())
        self.logger = logger
        self.benchmark_listener = (
            None
            if benchmarker is None
            else BenchmarkListener(benchmarker, manager.Queue(), logger)
        )
        self._helper = PipelineManagerHelper(
            pipeline, manager, self.log_listener, self.benchmark_listener
        )

    def start(self):
        """Starts the UI detection pipeline's worker processes"""
        logger = self.log_listener.get_logger()

        for name, module in self.pipeline.modules.items():
            self.workers[name] = self._create_worker(name, module, logger)

        self.log_listener.start()
        if self.benchmark_listener is not None:
            self.benchmark_listener.start()
        for worker in self.workers.values():
            worker.start()

    def _create_worker(
        self, name: PipelineModule, module: ModuleConstructor, logger: Logger
    ) -> Worker:
        """Creates a worker for the pipeline module

        Parameters
        ----------
        name : PipelineModule
            Pipeline module name.
        module : ModuleConstructor
            Constructor of the pipeline module.
        logger : Logger
            Logger to log the UI detection process from the worker process to be
            created.

        Returns
        -------
        Worker
            Handle of the worker process.
        """
        return Worker(
            module.func,
            self.pipeline.modules[name],
            self._helper.module_channels[name],
            self._helper.module_pools[name],
            logger,
            name,
        )

    def get_helper(self) -> PipelineHelper:
        """Gets a helper for accessing UI detection pipeline modules from a worker
        process

        Returns
        -------
        PipelineHelper
            Helper for accessing UI detection pipeline modules from a worker process.
        """
        return self._helper.get_helper()

    def terminate(self, force: bool = False):
        """Terminates the UI detection pipeline's worker processes

        Parameters
        ----------
        force : bool
            Whether to immediately terminate the worker process without waiting for
            the pending jobs to finish.
        """
        self.logger.info("Terminating the worker processes...")

        workers = list(self.workers.values()) + [self.log_listener]
        for worker in workers:
            worker.terminate(force)
            self.logger.info(f"'{worker.name}' worker has terminated.")

        optional_workers = (self.benchmark_listener,)
        for worker in optional_workers:
            if worker is not None:
                worker.terminate(force)
                self.logger.info(f"'{worker.name}' worker has terminated.")

        self._helper.close()  # Close the resources used by the pipeline.
