import numpy as np
from typing import Any, Callable, Dict, Iterable, List, Optional

from fluid_ai.base import UiDetectionModule, UiElement
from fluid_ai.icon import BaseIconLabeller
from fluid_ai.ocr import BaseOCR
from fluid_ai.ui.detection import BaseUiDetector
from fluid_ai.ui.matching import BaseUiMatching


class ModuleConstructor:
    constructor: Callable[..., UiDetectionModule]
    func: Callable
    args: Iterable
    kwargs: Dict

    def __init__(
        self,
        constructor: Callable[..., UiDetectionModule],
        *args,
        **kwargs,
    ):
        self.constructor = constructor
        self.args = args
        self.kwargs = kwargs

    def __call__(self) -> Any:
        return self.constructor(*self.args, **self.kwargs)


class PipelineConstructor:
    modules: Dict[str, ModuleConstructor]
    textual_elements: List[str]
    icon_elements: List[str]

    test_mode: bool

    DETECTOR = "detector"
    MATCHER = "matcher"
    TEXT_RECOGNIZER = "text_recognizer"
    ICON_LABELLER = "icon_labeller"

    def __init__(
        self,
        detector: ModuleConstructor,
        matcher: ModuleConstructor,
        text_recognizer: ModuleConstructor,
        icon_labeller: ModuleConstructor,
        textual_elements: List[str],
        icon_elements: List[str],
        test_mode: bool = False,
    ):
        detector.func = detect
        if test_mode:
            matcher.func = match_i
        else:
            matcher.func = match
        text_recognizer.func = recognize_texts
        icon_labeller.func = label_icons
        self.modules = {
            PipelineConstructor.DETECTOR: detector,
            PipelineConstructor.MATCHER: matcher,
            PipelineConstructor.TEXT_RECOGNIZER: text_recognizer,
            PipelineConstructor.ICON_LABELLER: icon_labeller,
        }
        self.textual_elements = textual_elements
        self.icon_elements = icon_elements
        self.test_mode = test_mode


def detect(
    _: int, screenshot: np.ndarray, module: UiDetectionModule
) -> List[UiElement]:
    assert isinstance(module, BaseUiDetector)
    return next(module([screenshot]))


def match(
    _: int,
    base: Optional[List[UiElement]],
    other: List[UiElement],
    module: UiDetectionModule,
) -> List[UiElement]:
    assert isinstance(module, BaseUiMatching)
    return module([] if base is None else base, other)


def match_i(
    job_no: int,
    base: Optional[List[UiElement]],
    other: List[UiElement],
    module: UiDetectionModule,
) -> List[UiElement]:
    assert isinstance(module, BaseUiMatching)
    base = [] if base is None else base
    return module.match_i(job_no, base, other)


def recognize_texts(
    _: int,
    elements: List[UiElement],
    module: UiDetectionModule,
) -> List[UiElement]:
    assert isinstance(module, BaseOCR)
    module(elements)
    return elements


def label_icons(
    _: int,
    elements: List[UiElement],
    module: UiDetectionModule,
) -> List[UiElement]:
    assert isinstance(module, BaseIconLabeller)
    module(elements)
    return elements
