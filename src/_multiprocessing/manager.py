import ctypes
import multiprocessing
import numpy as np
from multiprocessing.connection import Connection
from typing import Any, List, Tuple

import numpy as np
from fluid_ai.base import UiElement
from fluid_ai.icon import BaseIconLabeller
from fluid_ai.ocr import BaseOCR
from fluid_ai.pipeline import UiDetectionPipeline
from fluid_ai.ui.detection import BaseUiDetector

from src._multiprocessing.worker import Worker


class Screenshot:
    array: object
    shape: Tuple
    dtype: np.dtype

    def __init__(self, image: np.ndarray):
        # self.array = multiprocessing.RawArray(ctypes.c_double, image.flatten().tolist())
        self.array = image
        self.shape = image.shape
        self.dtype = image.dtype

    def to_numpy(self) -> np.ndarray:
        return self.array  # type: ignore
        np_array: np.ndarray = np.frombuffer(self.array, dtype=self.dtype)
        return np_array.reshape(self.shape)


def detect(screenshot: Screenshot, module: Any) -> List[UiElement]:
    assert isinstance(module, BaseUiDetector)
    return next(module.detect([screenshot.to_numpy()], save_img=False))


def recognize_texts(
    elements: List[UiElement],
    screenshot: Screenshot,
    module: Any,
) -> List[UiElement]:
    assert isinstance(module, BaseOCR)
    for e in elements:
        assert e.screenshot is None
    module.process(elements, screenshot.to_numpy)
    return elements


def label_icons(
    elements: List[UiElement],
    screenshot: Screenshot,
    module: Any,
) -> List[UiElement]:
    assert isinstance(module, BaseIconLabeller)
    for e in elements:
        assert e.screenshot is None
    module.process(elements, screenshot.to_numpy)
    return elements


class PipelineManager:
    pipeline: UiDetectionPipeline
    detector: BaseUiDetector
    text_recognizer: BaseOCR
    icon_labeller: BaseIconLabeller

    detector_pipe: Connection
    text_recognizer_pipe: Connection
    icon_labeller_pipe: Connection

    detector_worker: Worker
    text_recognizer_worker: Worker
    icon_labeller_worker: Worker

    def __init__(self, pipeline: UiDetectionPipeline):
        self.pipeline = pipeline
        self.detector = pipeline.detector
        self.text_recognizer = pipeline.text_recognizer
        self.icon_labeller = pipeline.icon_labeller

    def start(self):
        self.detector_pipe, detector_pipe = multiprocessing.Pipe()
        self.detector_worker = Worker(detect, detector_pipe, self.detector, "detector")
        self.text_recognizer_pipe, text_recognizer_pipe = multiprocessing.Pipe()
        self.text_recognizer_worker = Worker(
            recognize_texts,
            text_recognizer_pipe,
            self.text_recognizer,
            "text_recognizer",
        )
        self.icon_labeller_pipe, icon_labeller_pipe = multiprocessing.Pipe()
        self.icon_labeller_worker = Worker(
            label_icons, icon_labeller_pipe, self.icon_labeller, "icon_labeller"
        )

        print_open_fds()

        self.detector_worker.start()
        self.text_recognizer_worker.start()
        self.icon_labeller_worker.start()

    def terminate(self):
        self.detector_worker.terminate()
        self.text_recognizer_worker.terminate()
        self.icon_labeller_worker.terminate()


def print_open_fds(print_all=False):
    import os

    fds = set(os.listdir("/proc/self/fd/"))
    print(len(fds))
