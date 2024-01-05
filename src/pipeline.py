import logging
import os
import socket as sock
import sys
from abc import abstractmethod
from enum import Enum
from typing import List, Optional

import numpy as np

from fluid_ai.base import UiDetectionModule, UiElement
from fluid_ai.icon import BaseIconLabeler
from fluid_ai.ocr import BaseOCR
from fluid_ai.ui.detection import BaseUiDetector
from fluid_ai.ui.filter import BaseUiFilter
from fluid_ai.ui.matching import BaseUiMatching

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


class IPipelineServer:
    """
    An interface of a UI detection pipeline server.

    hostname : str
        Host name.
    port : str
        Port to listen to client connections.
    socket : Optional[sock.socket]
        Server socket.
    logger : ILogger
        Logger to log the UI detection process.
    """

    hostname: str
    port: str
    socket: Optional[sock.socket]
    logger: ILogger

    def __init__(
        self, hostname: str, port: str, socket: Optional[sock.socket], logger: ILogger
    ):
        self.hostname = hostname
        self.port = port
        self.socket = socket
        self.logger = logger

    @abstractmethod
    def start(self, _):
        """Starts the server"""
        raise NotImplementedError()

    def on_ready(self):
        pass

    def on_failure(self):
        pass

    def bind(self) -> sock.socket:
        socket = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
        socket.bind((self.hostname, self.port))
        socket.listen(1)
        self.logger.info(
            f'Pipeline server started serving at "{self.hostname}:{self.port} (PID={os.getpid()})".'
        )
        self.on_ready()
        return socket

    def exit(self, code: int):
        self.on_failure()
        sys.exit(code)

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
