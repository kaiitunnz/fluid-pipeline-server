from typing import Any, Callable, Dict, Iterable, List

from fluid_ai.base import UiDetectionModule

from src.pipeline import PipelineModule


class ModuleConstructor:
    constructor: Callable[..., UiDetectionModule]
    func: Callable
    is_thread: bool
    args: Iterable
    kwargs: Dict

    def __init__(
        self,
        constructor: Callable[..., UiDetectionModule],
        *args,
        is_thread: bool = True,
        **kwargs,
    ):
        self.constructor = constructor
        self.args = args
        self.is_thread = is_thread
        self.kwargs = kwargs

    def __call__(self) -> Any:
        return self.constructor(*self.args, **self.kwargs)


class PipelineConstructor:
    modules: Dict[PipelineModule, ModuleConstructor]
    textual_elements: List[str]
    icon_elements: List[str]

    test_mode: bool

    def __init__(
        self,
        detector: ModuleConstructor,
        filter: ModuleConstructor,
        matcher: ModuleConstructor,
        text_recognizer: ModuleConstructor,
        icon_labeler: ModuleConstructor,
        textual_elements: List[str],
        icon_elements: List[str],
        test_mode: bool = False,
    ):
        detector.func = PipelineModule.detect
        if test_mode:
            matcher.func = PipelineModule.match_i
        else:
            matcher.func = PipelineModule.match
        filter.func = PipelineModule.do_filter
        text_recognizer.func = PipelineModule.recognize_texts
        icon_labeler.func = PipelineModule.label_icons
        self.modules = {
            PipelineModule.DETECTOR: detector,
            PipelineModule.FILTER: filter,
            PipelineModule.MATCHER: matcher,
            PipelineModule.TEXT_RECOGNIZER: text_recognizer,
            PipelineModule.ICON_LABELER: icon_labeler,
        }
        self.textual_elements = textual_elements
        self.icon_elements = icon_elements
        self.test_mode = test_mode
