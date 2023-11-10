import os
from PIL import Image  # type: ignore
from queue import SimpleQueue
from typing import Callable, Dict, List, Optional, Tuple

from fluid_ai.base import UiElement

from src.constructor import ModuleConstructor, PipelineConstructor
from src.hybrid.benchmark import Benchmarker
from src.hybrid.logging import Logger
from src.hybrid.worker import Worker


class PipelineManagerHelper:
    key: int
    workers: Dict[str, Worker]

    textual_elements: List[str]
    icon_elements: List[str]

    logger: Logger
    benchmarker: Optional[Benchmarker]

    _count: int

    def __init__(
        self,
        key: int,
        constructor: PipelineConstructor,
        logger: Logger,
        benchmarker: Optional[Benchmarker],
    ):
        self.key = key
        self.workers = {}
        self.logger = logger
        self.benchmarker = benchmarker

        for name, module in constructor.modules.items():
            self.workers[name] = self._create_worker(name, module)

        self.textual_elements = constructor.textual_elements
        self.icon_elements = constructor.icon_elements

        self._count = 0

    def _create_worker(self, name: str, module: ModuleConstructor) -> Worker:
        return Worker(
            module.func, SimpleQueue(), module(), self.logger, name + str(self.key)
        )

    def get_helper(self) -> "PipelineHelper":
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
        for worker in self.workers.values():
            worker.start()

    def terminate(self, force: bool = False):
        for worker in self.workers.values():
            worker.terminate(force)


class PipelineHelper:
    key: int

    _workers: Dict[str, Worker]

    textual_elements: List[str]
    icon_elements: List[str]

    logger: Logger
    benchmarker: Optional[Benchmarker]

    _channel: SimpleQueue

    def __init__(
        self,
        key: int,
        workers: Dict[str, Worker],
        textual_elements: List[str],
        icon_elements: List[str],
        logger: Logger,
        benchmarker: Optional[Benchmarker],
    ):
        self.key = key
        self._workers = workers
        self.textual_elements = textual_elements
        self.icon_elements = icon_elements
        self.logger = logger
        self.benchmarker = benchmarker
        self._channel = SimpleQueue()

    def send(self, target: str, *args):
        self._workers[target].channel.put((self._channel, args))

    def wait_result(self) -> List[UiElement]:
        return self._channel.get()

    def save_image(self, img: Image.Image, save_dir: str, prefix: str = "img"):
        img.save(os.path.join(save_dir, f"{prefix}{self.key}.jpg"))

    def save_image_i(
        self, img: Image.Image, i: int, save_dir: str, prefix: str = "img"
    ):
        img.save(os.path.join(save_dir, f"{prefix}{i}.jpg"))

    @staticmethod
    def _log(log_func: Callable[[str], None], addr: Tuple[str, int], msg: str):
        ip, port = addr
        log_func(f"({ip}:{port}) {msg}")

    def log_debug(self, addr: Tuple[str, int], msg: str):
        self._log(self.logger.debug, addr, msg)

    def log_info(self, addr: Tuple[str, int], msg: str):
        self._log(self.logger.info, addr, msg)

    def log_warning(self, addr: Tuple[str, int], msg: str):
        self._log(self.logger.warn, addr, msg)

    def log_error(self, addr: Tuple[str, int], msg: str):
        self._log(self.logger.error, addr, msg)

    def log_critical(self, addr: Tuple[str, int], msg: str):
        self._log(self.logger.critical, addr, msg)
