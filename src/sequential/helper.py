from queue import SimpleQueue
from typing import List, Literal, Optional

from fluid_ai.base import UiElement

from src.benchmark import IBenchmarker
from src.logger import ILogger
from src.pipeline import IPipelineHelper, PipelineModule


class PipelineHelper(IPipelineHelper):
    """
    Helper for accessing the UI detection pipeline.

    Attributes
    ----------
    textual_elements : List[str]
        List of textual UI class names.
    icon_elements : List[str]
        List of icon UI class names.
    logger : Logger
        Logger to log the UI detection process.
    benchmarker : Optional[Benchmarker]
        Benchmarker to benchmark the UI detection pipeline server. `None` to not
        benchmark the server.
    """

    _channel: SimpleQueue
    _res_channel: SimpleQueue

    def __init__(
        self,
        pipeline_ch: SimpleQueue,
        logger: ILogger,
        benchmarker: Optional[IBenchmarker],
        textual_elements: List[str],
        icon_elements: List[str],
    ):
        """
        Parameters
        ----------
        pipeline_ch : SimpleQueue
            Channel on which the pipeline worker listens for new jobs.
        logger : Logger
            Logger to log the UI detection process.
        benchmarker : Optional[Benchmarker]
            Benchmarker to benchmark the UI detection pipeline server. `None` to not
            benchmark the server.
        textual_elements : List[str]
            List of textual UI class names.
        icon_elements : List[str]
            List of icon UI class names.
        """
        super().__init__(logger, benchmarker, textual_elements, icon_elements)
        self._channel = pipeline_ch
        self._res_channel = SimpleQueue()

    def send(self, _: Literal[PipelineModule.DETECTOR], *args):
        self._channel.put((self._res_channel, args))

    def wait(self, _: Literal[PipelineModule.DETECTOR]) -> List[UiElement]:
        return self._res_channel.get()
