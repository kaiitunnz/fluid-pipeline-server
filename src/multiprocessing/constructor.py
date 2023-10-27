from typing import Any, Callable, Dict, Iterable, List

from fluid_ai.icon import BaseIconLabeller
from fluid_ai.ocr import BaseOCR
from fluid_ai.ui.detection import BaseUiDetector


class PipelineConstructor:
    detector: Callable[[], BaseUiDetector]
    text_recognizer: Callable[[], BaseOCR]
    icon_labeller: Callable[[], BaseIconLabeller]
    textual_elements: List[str]
    icon_elements: List[str]

    def __init__(
        self,
        detector: Callable[[], BaseUiDetector],
        text_recognizer: Callable[[], BaseOCR],
        icon_labeller: Callable[[], BaseIconLabeller],
        textual_elements: List[str],
        icon_elements: List[str],
    ):
        self.detector = detector
        self.text_recognizer = text_recognizer
        self.icon_labeller = icon_labeller
        self.textual_elements = textual_elements
        self.icon_elements = icon_elements


class ModuleConstructor:
    func: Callable[..., Any]
    args: Iterable
    kwargs: Dict

    def __init__(self, func: Callable[..., Any], *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self) -> Any:
        return self.func(*self.args, **self.kwargs)
