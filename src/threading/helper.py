import os
from PIL import Image  # type: ignore
from logging import Logger
from queue import SimpleQueue
from typing import Callable, Dict, List, Optional, Tuple

from fluid_ai.base import UiElement

from src.benchmark import Benchmarker
from src.constructor import ModuleConstructor, PipelineConstructor
from src.pipeline import PipelineModule
from src.threading.loadbalancer import LoadBalancer
from src.threading.worker import Worker


class PipelineManagerHelper:
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
        self.channels = {}
        self.num_instances = num_instances
        self.logger = logger
        self.benchmarker = benchmarker

        for name, module in constructor.modules.items():
            self.channels[name] = self._create_module_instance(name, module)

        self.textual_elements = constructor.textual_elements
        self.icon_elements = constructor.icon_elements

        self._count = 0

    def _create_module_instance(
        self, name: PipelineModule, module: ModuleConstructor
    ) -> LoadBalancer:
        workers = [
            Worker(
                module.func, SimpleQueue(), module(), self.logger, name.value + str(i)
            )
            for i in range(self.num_instances)
        ]
        return LoadBalancer(workers, self.logger)

    def get_helper(self) -> "PipelineHelper":
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

    def start(self):
        for balancer in self.channels.values():
            balancer.start()

    def terminate(self):
        for balancer in self.channels.values():
            balancer.terminate()


class PipelineHelper:
    key: int

    _channels: Dict[PipelineModule, LoadBalancer]

    textual_elements: List[str]
    icon_elements: List[str]

    logger: Logger
    benchmarker: Optional[Benchmarker]

    _channel: SimpleQueue

    def __init__(
        self,
        key: int,
        channels: Dict[PipelineModule, LoadBalancer],
        textual_elements: List[str],
        icon_elements: List[str],
        logger: Logger,
        benchmarker: Optional[Benchmarker],
    ):
        self.key = key
        self._channels = channels
        self.textual_elements = textual_elements
        self.icon_elements = icon_elements
        self.logger = logger
        self.benchmarker = benchmarker
        self._channel = SimpleQueue()

    def send(self, target: PipelineModule, *args):
        self._channels[target].send((self._channel, args))

    def sendi(self, target: PipelineModule, i: int, *args):
        self._channels[target].sendi(i, (self._channel, args))

    def wait_result(self) -> List[UiElement]:
        return self._channel.get()

    def save_image(self, img: Image.Image, save_dir: str, prefix: str = "img"):
        img.save(os.path.join(save_dir, f"{prefix}{self.key}.jpg"))

    def get_num_instances(self) -> int:
        if len(self._channels) == 0:
            return 0
        return len(tuple(self._channels.values())[0].workers)

    @staticmethod
    def _log(log_func: Callable[[str], None], addr: Tuple[str, int], msg: str):
        ip, port = addr
        log_func(f"({ip}:{port}) {msg}")

    def log_debug(self, addr: Tuple[str, int], msg: str):
        PipelineHelper._log(self.logger.debug, addr, msg)

    def log_info(self, addr: Tuple[str, int], msg: str):
        PipelineHelper._log(self.logger.info, addr, msg)

    def log_warning(self, addr: Tuple[str, int], msg: str):
        PipelineHelper._log(self.logger.warn, addr, msg)

    def log_error(self, addr: Tuple[str, int], msg: str):
        PipelineHelper._log(self.logger.error, addr, msg)

    def log_critical(self, addr: Tuple[str, int], msg: str):
        PipelineHelper._log(self.logger.critical, addr, msg)
