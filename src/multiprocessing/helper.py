import os
from PIL import Image
from queue import Queue
from typing import Callable, List, Tuple

from fluid_ai.base import UiElement
from multiprocessing.managers import DictProxy, SyncManager
from threading import Condition
from torch import Tensor

from src.multiprocessing.constructor import PipelineConstructor
from src.multiprocessing.logging import LogListener, Logger


class PipelineManagerHelper:
    manager: SyncManager
    log_listener: LogListener

    detector_ch: Queue
    text_recognizer_ch: Queue
    icon_labeller_ch: Queue

    detector_pool: DictProxy
    text_recognizer_pool: DictProxy
    icon_labeller_pool: DictProxy

    textual_elements: List[str]
    icon_elements: List[str]

    _count: int
    _conditions: DictProxy

    def __init__(
        self,
        pipeline: PipelineConstructor,
        manager: SyncManager,
        log_listener: LogListener,
    ):
        self.manager = manager
        self.log_listener = log_listener

        self.detector_ch = manager.Queue()
        self.text_recognizer_ch = manager.Queue()
        self.icon_labeller_ch = manager.Queue()

        self.detector_pool = manager.dict()
        self.text_recognizer_pool = manager.dict()
        self.icon_labeller_pool = manager.dict()

        self.textual_elements = pipeline.textual_elements
        self.icon_elements = pipeline.icon_elements

        self._count = 0
        self._conditions = manager.dict()

    def clone(self) -> "PipelineHelper":
        key = self._count
        self._count += 1

        condition = self.manager.Condition()
        self._conditions[key] = condition

        return PipelineHelper(
            key,
            self.detector_ch,
            self.text_recognizer_ch,
            self.icon_labeller_ch,
            self.detector_pool,
            self.text_recognizer_pool,
            self.icon_labeller_pool,
            self.textual_elements,
            self.icon_elements,
            self.log_listener.get_logger(),
            condition,
            self._conditions,
        )


class PipelineHelper:
    key: int
    detector_ch: Queue
    text_recognizer_ch: Queue
    icon_labeller_ch: Queue

    detector_pool: DictProxy
    text_recognizer_pool: DictProxy
    icon_labeller_pool: DictProxy

    textual_elements: List[str]
    icon_elements: List[str]

    logger: Logger

    condition: Condition
    manager_conditions: DictProxy

    def __init__(
        self,
        key: int,
        detector_ch: Queue,
        text_recognizer_ch: Queue,
        icon_labeller_ch: Queue,
        detector_pool: DictProxy,
        text_recognizer_pool: DictProxy,
        icon_labeller_pool: DictProxy,
        textual_elements: List[str],
        icon_elements: List[str],
        logger: Logger,
        condition: Condition,
        manager_conditions: DictProxy,
    ):
        self.key = key
        self.detector_ch = detector_ch
        self.text_recognizer_ch = text_recognizer_ch
        self.icon_labeller_ch = icon_labeller_ch
        self.detector_pool = detector_pool
        self.text_recognizer_pool = text_recognizer_pool
        self.icon_labeller_pool = icon_labeller_pool
        self.textual_elements = textual_elements
        self.icon_elements = icon_elements
        self.logger = logger
        self.condition = condition
        self.manager_conditions = manager_conditions

    def detect(self, screenshot: Tensor):
        self.detector_ch.put((self.key, self.condition, (screenshot,)))

    def wait_detect(self) -> List[UiElement]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.detector_pool.get(self.key, None) is not None
            )
            return self.detector_pool.pop(self.key)

    def recognize_text(self, elements: List[UiElement]):
        self.text_recognizer_ch.put((self.key, self.condition, (elements,)))

    def wait_recognize_text(self) -> List[UiElement]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.text_recognizer_pool.get(self.key, None) is not None
            )
            return self.text_recognizer_pool.pop(self.key)

    def label_icons(self, elements: List[UiElement]):
        self.icon_labeller_ch.put((self.key, self.condition, (elements,)))

    def wait_label_icons(self) -> List[UiElement]:
        with self.condition:
            self.condition.wait_for(
                lambda: self.icon_labeller_pool.get(self.key, None) is not None
            )
            return self.icon_labeller_pool.pop(self.key)

    def save_image(self, img: Image.Image, save_dir: str, prefix: str = "img"):
        img.save(os.path.join(save_dir, f"{prefix}{self.key}.jpg"))

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

    def __del__(self):
        self.manager_conditions.pop(self.key, None)
