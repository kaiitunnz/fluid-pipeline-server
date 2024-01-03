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


class PipelineServerInterface:
    @abstractmethod
    def start(self, _):
        raise NotImplementedError()


class PipelineModule(Enum):
    DETECTOR = "detector"
    FILTER = "filter"
    MATCHER = "matcher"
    TEXT_RECOGNIZER = "text_recognizer"
    ICON_LABELER = "icon_labeler"

    @staticmethod
    def detect(
        _: int, screenshot: np.ndarray, module: UiDetectionModule
    ) -> List[UiElement]:
        assert isinstance(module, BaseUiDetector)
        return next(module([screenshot]))

    @staticmethod
    def do_filter(
        _: int, elements: List[UiElement], module: UiDetectionModule
    ) -> List[UiElement]:
        assert isinstance(module, BaseUiFilter)
        return module(elements)

    @staticmethod
    def match(
        _: int,
        base: Optional[List[UiElement]],
        other: List[UiElement],
        module: UiDetectionModule,
    ) -> List[UiElement]:
        assert isinstance(module, BaseUiMatching)
        return module([] if base is None else base, other)

    @staticmethod
    def match_i(
        job_no: int,
        base: Optional[List[UiElement]],
        other: List[UiElement],
        module: UiDetectionModule,
    ) -> List[UiElement]:
        assert isinstance(module, BaseUiMatching)
        base = [] if base is None else base
        return module.match_i(job_no, base, other)

    @staticmethod
    def recognize_texts(
        _: int,
        elements: List[UiElement],
        module: UiDetectionModule,
    ) -> List[UiElement]:
        assert isinstance(module, BaseOCR)
        module(elements)
        return elements

    @staticmethod
    def label_icons(
        _: int,
        elements: List[UiElement],
        module: UiDetectionModule,
    ) -> List[UiElement]:
        assert isinstance(module, BaseIconLabeler)
        module(elements)
        return elements
