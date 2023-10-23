import logging
from queue import SimpleQueue
from typing import Any, List

import numpy as np
from fluid_ai.base import UiElement
from fluid_ai.pipeline import UiDetectionPipeline

from src.threading.worker import Worker


def detect(screenshot: np.ndarray, module: Any) -> List[UiElement]:
    assert isinstance(module, UiDetectionPipeline)
    return next(module.detect([screenshot]))


class PipelineManager:
    pipeline: UiDetectionPipeline
    pipeline_ch: SimpleQueue
    worker: Worker
    logger: logging.Logger

    def __init__(self, pipeline: UiDetectionPipeline, logger: logging.Logger):
        self.pipeline = pipeline
        self.pipeline_ch = SimpleQueue()
        self.logger = logger

    def start(self):
        self.worker = Worker(
            detect, self.pipeline_ch, self.pipeline, self.logger, "pipeline"
        )
        self.worker.start()

    def terminate(self):
        self.worker.terminate()
