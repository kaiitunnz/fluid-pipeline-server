from typing import Any, Callable, Dict, Iterable, List

from fluid_ai.base import UiDetectionModule

from src.pipeline import PipelineModule


class ModuleConstructor:
    """
    A lazy constructor of a UI detection pipeline module.

    Attributes
    ----------
    constructor : Callable[..., UiDetectionModule]
        Constructor of a UI detection pipeline module.
    func : Callable
        Function that utilizes the constructed module.
    is_thread : bool
        Whether to spawns a new thread to execute the module.
    args : Iterable
        Positional arguments to the constructor.
    kwargs : Dict
        Keyword arguments to the constructor.
    """

    constructor: Callable[..., UiDetectionModule]
    func: Callable
    is_thread: bool
    args: Iterable
    kwargs: Dict[str, Any]

    def __init__(
        self,
        constructor: Callable[..., UiDetectionModule],
        *args,
        is_thread: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        constructor : Callable[..., UiDetectionModule]
            Constructor of a UI detection pipeline module.
        *args
            Positional arguments to the constructor.
        is_thread : bool
            Whether to spawns a new thread to execute the module.
        **kwargs
            Keyword arguments to the constructor.
        """
        self.constructor = constructor
        self.args = args
        self.is_thread = is_thread
        self.kwargs = kwargs

    def __call__(self) -> Any:
        """Initializes the UI detection pipeline module from the constructor

        Returns
        -------
        Any
            Initialized UI detection pipeline module.
        """
        return self.constructor(*self.args, **self.kwargs)


class PipelineConstructor:
    """
    A lazy constructor of a UI detection pipeline.

    Attributes
    ----------
    modules : Dict[PipelineModule, ModuleConstructor]
        Mapping from pipeline module names to module constructors.
    textual_elements : List[str]
        List of textual UI class names.
    icon_elements : List[str]
        List of icon UI class names.
    test_mode : bool
        Whether to initializes the pipeline in test mode.
    """

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
        relation: ModuleConstructor,
        textual_elements: List[str],
        icon_elements: List[str],
        test_mode: bool = False,
    ):
        """
        Parameters
        ----------
        detector : ModuleConstructor
            Constructor for a UI detection model.
        filter : ModuleConstructor
            Constructor for a UI filter, aka an invalid UI detection model.
        matcher : ModuleConstructor
            Constructor for a UI matching model.
        text_recognizer : ModuleConstructor
            Constructor for a text recognition module.
        icon_labeler : ModuleConstructor
            Constructor for an icon labeling module.
        relation: ModuleConstructor
            Constructor for a UI relation module.
        textual_elements : List[str]
            List of textual UI class names.
        icon_elements : List[str]
            List of icon UI class names.
        test_mode : bool
            Whether to initializes the pipeline in test mode.
        """
        detector.func = PipelineModule.detect
        if test_mode:
            matcher.func = PipelineModule.match_i
        else:
            matcher.func = PipelineModule.match
        filter.func = PipelineModule.do_filter
        text_recognizer.func = PipelineModule.recognize_texts
        icon_labeler.func = PipelineModule.label_icons
        relation.func = PipelineModule.relate
        self.modules = {
            PipelineModule.DETECTOR: detector,
            PipelineModule.FILTER: filter,
            PipelineModule.MATCHER: matcher,
            PipelineModule.TEXT_RECOGNIZER: text_recognizer,
            PipelineModule.ICON_LABELER: icon_labeler,
            PipelineModule.RELATION: relation,
        }
        self.textual_elements = textual_elements
        self.icon_elements = icon_elements
        self.test_mode = test_mode
