import json
import os
import sys
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Callable, Optional

import torch

from fluid_ai.icon import BaseIconLabeler, ClassifierIconLabeler, DummyIconLabeler
from fluid_ai.ocr import BaseOCR, DummyOCR, EasyOCR
from fluid_ai.pipeline import UiDetectionPipeline
from fluid_ai.ui.detection import YoloUiDetector
from fluid_ai.ui.filter import (
    BaseUiFilter,
    BoundaryUiFilter,
    DummyUiFilter,
    ElementUiFilter,
)
from fluid_ai.ui.matching import (
    BaseUiMatching,
    IouUiMatching,
    HogUiMatching,
    GistUiMatching,
)

from src.constructor import ModuleConstructor, PipelineConstructor
from src.hybrid.server import PipelineServer as HybridServer
from src.multiprocess.server import PipelineServer as MultiprocessServer
from src.multithread.server import PipelineServer as MultithreadServer
from src.pipeline import IPipelineServer
from src.sequential.server import PipelineServer as SequentialServer


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument(
        "-b",
        "--benchmark",
        action="store",
        help="Path to the benchmark result file",
        default=None,
    )
    parser.add_argument(
        "-l", "--log", action="store", help="Path to the log file", default=None
    )
    parser.add_argument("-s", "--silent", action="store_true", default=False)
    parser.add_argument(
        "-m",
        "--mode",
        action="store",
        choices=["sequential", "multithread", "multiprocess", "hybrid"],
        default="hybrid",
        help="Pipeline server mode",
    )
    return parser.parse_args()


def pipeline_from_config(config: Dict[str, Any]) -> UiDetectionPipeline:
    detector = YoloUiDetector(
        config["ui_detector"]["path"],
        device=torch.device(config["ui_detector"]["device"]),
    )

    filter: BaseUiFilter
    if config["ui_filter"]["dummy"]:
        filter = DummyUiFilter()
    else:
        filter_method = config["ui_filter"]["method"].lower()
        if filter_method == "element":
            filter = ElementUiFilter(**config["ui_filter"]["args"])
        elif filter_method == "boundary":
            filter = BoundaryUiFilter(**config["ui_filter"]["args"])
        else:
            raise NotImplementedError("The filter method is not implemented")

    matching_method = config["ui_matcher"]["method"].lower()
    matcher: BaseUiMatching
    if matching_method == "iou":
        matcher = IouUiMatching(**config["ui_matcher"]["args"])
    elif matching_method == "gist":
        matcher = GistUiMatching(**config["ui_matcher"]["args"])
    elif matching_method == "hog":
        matcher = HogUiMatching(**config["ui_matcher"]["args"])
    else:
        raise NotImplementedError("The matching method is not implemented")

    text_recognizer: BaseOCR
    if config["text_recognizer"]["dummy"]:
        text_recognizer = DummyOCR()
    else:
        text_recognizer = EasyOCR(batch_size=config["text_recognizer"]["batch_size"])

    icon_labeler: BaseIconLabeler
    if config["icon_labeler"]["dummy"]:
        icon_labeler = DummyIconLabeler()
    else:
        icon_labeler = ClassifierIconLabeler(**config["icon_labeler"]["args"])

    return UiDetectionPipeline(
        detector,
        filter,
        matcher,
        text_recognizer,
        config["special_elements"]["text"],
        icon_labeler,
        config["special_elements"]["icon"],
    )


def constructor_from_config(config: Dict[str, Any]) -> PipelineConstructor:
    detector = ModuleConstructor(
        YoloUiDetector,
        config["ui_detector"]["path"],
        device=torch.device(config["ui_detector"]["device"]),
    )

    if config["ui_filter"]["dummy"]:
        filter = ModuleConstructor(DummyUiFilter)
    else:
        filter_method = config["ui_filter"]["method"].lower()
        if filter_method == "element":
            filter = ModuleConstructor(ElementUiFilter, **config["ui_filter"]["args"])
        elif filter_method == "boundary":
            filter = ModuleConstructor(BoundaryUiFilter, **config["ui_filter"]["args"])
        else:
            raise NotImplementedError("The filter method is not implemented")

    matching_method = config["ui_matcher"]["method"].lower()
    if matching_method == "iou":
        matcher = ModuleConstructor(
            IouUiMatching,
            is_thread=config["ui_matcher"]["is_thread"],
            **config["ui_matcher"]["args"],
        )
    elif matching_method == "gist":
        matcher = ModuleConstructor(
            GistUiMatching,
            is_thread=config["ui_matcher"]["is_thread"],
            **config["ui_matcher"]["args"],
        )
    elif matching_method == "hog":
        matcher = ModuleConstructor(
            HogUiMatching,
            is_thread=config["ui_matcher"]["is_thread"],
            **config["ui_matcher"]["args"],
        )
    else:
        raise NotImplementedError("The matching method is not implemented")

    if config["text_recognizer"]["dummy"]:
        text_recognizer = ModuleConstructor(DummyOCR)
    else:
        text_recognizer = ModuleConstructor(
            EasyOCR, batch_size=config["text_recognizer"]["batch_size"]
        )

    if config["icon_labeler"]["dummy"]:
        icon_labeler = ModuleConstructor(DummyIconLabeler)
    else:
        icon_labeler = ModuleConstructor(
            ClassifierIconLabeler,
            **config["icon_labeler"]["args"],
        )

    return PipelineConstructor(
        detector,
        filter,
        matcher,
        text_recognizer,
        icon_labeler,
        config["special_elements"]["text"],
        config["special_elements"]["icon"],
        test_mode=config["test_mode"],
    )


def init_pipeline_server(
    config: Dict[str, Any],
    mode: str,
    verbose: bool,
    benchmark: Optional[str],
    on_failure: Optional[Callable[[], Any]] = None,
) -> IPipelineServer:
    pipeline_server: IPipelineServer
    if mode == "sequential":
        try:
            pipeline = pipeline_from_config(config)
        except Exception as e:
            if on_failure is not None:
                on_failure()
            raise e
        pipeline_server = SequentialServer(
            **config["server"],
            pipeline=pipeline,
            verbose=verbose,
            benchmark=benchmark is not None,
            benchmark_file=benchmark,
            test_mode=config["test_mode"],
        )
    elif mode == "multithread":
        constructor = constructor_from_config(config)
        pipeline_server = MultithreadServer(
            **config["server"],
            pipeline=constructor,
            verbose=verbose,
            benchmark=benchmark is not None,
            benchmark_file=benchmark,
        )
    elif mode == "multiprocess":
        constructor = constructor_from_config(config)
        pipeline_server = MultiprocessServer(
            **config["server"],
            pipeline=constructor,
            verbose=verbose,
            benchmark=benchmark is not None,
            benchmark_file=benchmark,
        )
    elif mode == "hybrid":
        constructor = constructor_from_config(config)
        pipeline_server = HybridServer(
            **config["server"],
            pipeline=constructor,
            verbose=verbose,
            benchmark=benchmark is not None,
            benchmark_file=benchmark,
        )
    else:
        raise NotImplementedError()
    return pipeline_server


def main(args: Namespace):
    with open("config.json", "r") as f:
        config = json.load(f)
    sample_file = config["server"].pop("sample_file")
    pipeline_server = init_pipeline_server(
        config, args.mode, args.verbose, args.benchmark
    )
    pipeline_server.start(sample_file)


if __name__ == "__main__":
    args = parse_args()
    log_path: Optional[str]
    if args.silent:
        log_path = os.path.devnull
    elif args.log is None:
        log_path = None
    else:
        log_path = args.log

    if log_path is None:
        main(args)
    else:
        with open(log_path, "w") as log:
            sys.stderr = log
            sys.stdout = log
            main(args)
