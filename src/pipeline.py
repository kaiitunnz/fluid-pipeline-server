import logging
import os
from PIL import Image  # type: ignore
from abc import abstractmethod
from enum import Enum
from typing import Callable, List, Optional, Tuple

import numpy as np

from fluid_ai.base import UiDetectionModule, UiElement
from fluid_ai.icon import BaseIconLabeler
from fluid_ai.ocr import BaseOCR
from fluid_ai.ui.detection import BaseUiDetector
from fluid_ai.ui.filter import BaseUiFilter
from fluid_ai.ui.matching import BaseUiMatching

from src.benchmark import IBenchmarker
from src.logger import ILogger


class PipelineModule(Enum):
    """
    An enum representing names of UI detection pipeline modules.
    """

    DETECTOR = "detector"
    """
    UI detection model.
    """
    FILTER = "filter"
    """
    UI filter model, aka invalid UI detection model.
    """
    MATCHER = "matcher"
    """
    UI matching model.
    """
    TEXT_RECOGNIZER = "text_recognizer"
    """
    Text recognition module.
    """
    ICON_LABELER = "icon_labeler"
    """
    Icon labeling module.
    """

    @staticmethod
    def detect(
        _: int, screenshot: np.ndarray, module: UiDetectionModule
    ) -> List[UiElement]:
        """Detects UI elements from a screenshot

        Parameters
        ----------
        screenshot : ndarray
            Screenshot to be processed.
        module : UiDetectionModule
            UI detection pipeline module. Must inherit `BaseUiDetector`.

        Returns
        -------
        List[UiElement]
            List of detected UI elements.
        """
        assert isinstance(module, BaseUiDetector)
        return next(module([screenshot]))

    @staticmethod
    def do_filter(
        _: int, elements: List[UiElement], module: UiDetectionModule
    ) -> List[UiElement]:
        """Filters valid UI elements

        Parameters
        ----------
        elements : List[UiElement]
            UI elements to be filtered.
        module : UiDetectionModule
            UI detection pipeline module. Must inherit `BaseUiFilter`.

        Returns
        -------
        List[UiElement]
            List of filtered UI elements.
        """
        assert isinstance(module, BaseUiFilter)
        return module(elements)

    @staticmethod
    def match(
        _: int,
        base: Optional[List[UiElement]],
        other: List[UiElement],
        module: UiDetectionModule,
    ) -> List[UiElement]:
        """Merge two lists of UI elements

        UI elements in `base` are matched with those in `other`. For each pair of
        matching elements, that from `base` will be included in the resuling list
        while the other will be discarded.

        Parameters
        ----------
        base: Optional[List[UiElement]]
            List of base UI elements. `None` will be taken as an empty list.
        other : List[UiElement]
            List of additional UI elements.
        module : UiDetectionModule
            UI detection pipeline module. Must inherit `BaseUiMatching`.

        Returns
        -------
        List[UiElement]
            Merged list of UI elements.
        """
        assert isinstance(module, BaseUiMatching)
        return module([] if base is None else base, other)

    @staticmethod
    def match_i(
        job_no: int,
        base: Optional[List[UiElement]],
        other: List[UiElement],
        module: UiDetectionModule,
    ) -> List[UiElement]:
        """Merge two lists of UI elements

        Similar to the `PipelineModule.match()` method but utilizes the job number
        for debugging or benchmarking.

        Parameters
        ----------
        job_no : int
            Number of the current job.
        base: Optional[List[UiElement]]
            List of base UI elements. `None` will be taken as an empty list.
        other : List[UiElement]
            List of additional UI elements.
        module : UiDetectionModule
            UI detection pipeline module. Must inherit `BaseUiMatching`.

        Returns
        -------
        List[UiElement]
            Merged list of UI elements.
        """
        assert isinstance(module, BaseUiMatching)
        base = [] if base is None else base
        return module.match_i(job_no, base, other)

    @staticmethod
    def recognize_texts(
        _: int,
        elements: List[UiElement],
        module: UiDetectionModule,
    ) -> List[UiElement]:
        """Recognizes texts in the UI elements

        The recognized text of each UI element, `element`, is stored with a "text"
        key in `element.info`.

        Parameters
        ----------
        elements : List[UiElement]
            List of UI elements to be processed. The elements should be of textual
            UI classes.
        module : UiDetectionModule
            UI detection pipeline module. Must inherit `BaseOCR`.

        Returns
        -------
        List[UiElement]
            Resulting list of UI elements.
        """
        assert isinstance(module, BaseOCR)
        module(elements)
        return elements

    @staticmethod
    def label_icons(
        _: int,
        elements: List[UiElement],
        module: UiDetectionModule,
    ) -> List[UiElement]:
        """Generates icon labels for the UI elements

        The icon label of each UI element, `element`, is stored with an "icon_label"
        key in `element.info`.

        Parameters
        ----------
        elements : List[UiElement]
            List of UI elements to be processed. The elements should be of icon UI
            classes.
        module : UiDetectionModule
            UI detection pipeline module. Must inherit `BaseIconLabeler`.

        Returns
        -------
        List[UiElement]
            Resulting list of UI elements.
        """
        assert isinstance(module, BaseIconLabeler)
        module(elements)
        return elements


class IPipelineHelper:
    """
    Helper for accessing the UI detection pipeline modules.

    Attributes
    ----------
    logger : ILogger
        Logger to log the UI detection process.
    benchmarker : Optional[IBenchmarker]
        Benchmarker to benchmark the UI detection pipeline server. `None` to not
        benchmark the server.
    textual_elements : List[str]
        List of textual UI class names.
    icon_elements : List[str]
        List of icon UI class names.
    """

    logger: ILogger
    benchmarker: Optional[IBenchmarker]
    textual_elements: List[str]
    icon_elements: List[str]

    def __init__(
        self,
        logger: ILogger,
        benchmarker: Optional[IBenchmarker],
        textual_elements: List[str],
        icon_elements: List[str],
    ):
        """
        Attributes
        ----------
        logger : ILogger
            Logger to log the UI detection process.
        benchmarker : Optional[IBenchmarker]
            Benchmarker to benchmark the UI detection pipeline server. `None` to not
            benchmark the server.
        textual_elements : List[str]
            List of textual UI class names.
        icon_elements : List[str]
            List of icon UI class names.
        """
        self.logger = logger
        self.benchmarker = benchmarker
        self.textual_elements = textual_elements
        self.icon_elements = icon_elements

    @abstractmethod
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
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    @staticmethod
    def save_image_i(img: Image.Image, i: int, save_dir: str, prefix: str = "img"):
        """Saves an image with a filename containing `prefix` and `i`

        Parameters
        ----------
        img : Image
            The image to be saved.
        i : int
            Integer to be included in the file name.
        save_dir : str
            Path to the directory where the image will be saved.
        prefix : str
            File name's prefix. The saved file name will be `prefix` followed by `i`.
        """
        img.save(os.path.join(save_dir, f"{prefix}{i}.jpg"))

    @staticmethod
    def _log(log_func: Callable[[str], None], addr: Tuple[str, int], msg: str):
        """Formats a message with the associated client's IP and port and logs it
        with the given logging function

        Parameters
        ----------
        log_func : Callable[[str], None]
            Logging function that takes a string to be logged.
        addr : Tuple[str, int]
            `(IP address, port)` of the associated client.
        msg : str
            Actual message to be logged.
        """
        ip, port = addr
        log_func(f"({ip}:{port}) {msg}")

    def log_debug(self, addr: Tuple[str, int], msg: str):
        """Formats a message with the associated client's IP and port and logs it in
        the debug level

        Parameters
        ----------
        addr : Tuple[str, int]
            `(IP address, port)` of the associated client.
        msg : str
            Actual message to be logged.
        """
        self._log(self.logger.debug, addr, msg)

    def log_info(self, addr: Tuple[str, int], msg: str):
        """Formats a message with the associated client's IP and port and logs it in
        the info level

        Parameters
        ----------
        addr : Tuple[str, int]
            `(IP address, port)` of the associated client.
        msg : str
            Actual message to be logged.
        """
        self._log(self.logger.info, addr, msg)

    def log_warning(self, addr: Tuple[str, int], msg: str):
        """Formats a message with the associated client's IP and port and logs it in
        the warning level

        Parameters
        ----------
        addr : Tuple[str, int]
            `(IP address, port)` of the associated client.
        msg : str
            Actual message to be logged.
        """
        self._log(self.logger.warn, addr, msg)

    def log_error(self, addr: Tuple[str, int], msg: str):
        """Formats a message with the associated client's IP and port and logs it in
        the error level

        Parameters
        ----------
        addr : Tuple[str, int]
            `(IP address, port)` of the associated client.
        msg : str
            Actual message to be logged.
        """
        self._log(self.logger.error, addr, msg)

    def log_critical(self, addr: Tuple[str, int], msg: str):
        """Formats a message with the associated client's IP and port and logs it in
        the critical level

        Parameters
        ----------
        addr : Tuple[str, int]
            `(IP address, port)` of the associated client.
        msg : str
            Actual message to be logged.
        """
        self._log(self.logger.critical, addr, msg)


class IPipelineServer:
    """
    An interface of a UI detection pipeline server.
    """

    @abstractmethod
    def start(self, _):
        """Starts the server"""
        raise NotImplementedError()

    @classmethod
    def _init_logger(cls, verbose: bool = True) -> logging.Logger:
        """Initializes the logger to log server events

        Parameters
        ----------
        verbose : bool
            Whether to log server events verbosely.

        Returns
        -------
        Logger
            Initialized logger.
        """
        fmt = "[%(asctime)s | %(name)s] [%(levelname)s] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        logging.basicConfig(format=fmt, datefmt=datefmt)
        logger = logging.getLogger(cls.__name__)
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        return logger
