import json
import os
import sys
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Optional

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
from fluid_ai.ui.relation import BaseUiRelation, DummyUiRelation, UiOverlap

from src.constructor import ModuleConstructor, PipelineConstructor
from src.hybrid.server import PipelineServer as HybridServer
from src.multiprocess.server import PipelineServer as MultiprocessServer
from src.multithread.server import PipelineServer as MultithreadServer
from src.sequential.server import PipelineServer as SequentialServer
from src.server import IPipelineServer, ServerCallbacks


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

    relation: BaseUiRelation
    if config["ui_relation"]["dummy"]:
        relation = DummyUiRelation()
    else:
        relation_method = config["ui_relation"]["method"].lower()
        if relation_method == "overlap":
            relation = UiOverlap(**config["ui_relation"]["args"])
        else:
            raise NotImplementedError("The UI relation method is not implemented")

    return UiDetectionPipeline(
        detector,
        filter,
        matcher,
        text_recognizer,
        config["special_elements"]["text"],
        icon_labeler,
        config["special_elements"]["icon"],
        relation,
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

    if config["ui_relation"]["dummy"]:
        relation = ModuleConstructor(DummyUiRelation)
    else:
        relation_method = config["ui_relation"]["method"].lower()
        if relation_method == "overlap":
            relation = ModuleConstructor(
                UiOverlap,
                is_thread=config["ui_relation"]["is_thread"],
                **config["ui_relation"]["args"],
            )
        else:
            raise NotImplementedError("The UI relation method is not implemented")

    return PipelineConstructor(
        detector,
        filter,
        matcher,
        text_recognizer,
        icon_labeler,
        relation,
        config["special_elements"]["text"],
        config["special_elements"]["icon"],
        test_mode=config["test_mode"],
    )


def init_pipeline_server(
    config: Dict[str, Any],
    mode: str,
    verbose: bool,
    benchmark: Optional[str],
    callbacks: ServerCallbacks,
) -> IPipelineServer:
    pipeline_server: IPipelineServer
    if mode == "sequential":
        try:
            pipeline = pipeline_from_config(config)
        except Exception as e:
            callbacks.on_failure()
            raise e
        pipeline_server = SequentialServer(
            **config["server"],
            pipeline=pipeline,
            callbacks=callbacks,
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
            callbacks=callbacks,
            verbose=verbose,
            benchmark=benchmark is not None,
            benchmark_file=benchmark,
        )
    elif mode == "multiprocess":
        constructor = constructor_from_config(config)
        pipeline_server = MultiprocessServer(
            **config["server"],
            pipeline=constructor,
            callbacks=callbacks,
            verbose=verbose,
            benchmark=benchmark is not None,
            benchmark_file=benchmark,
        )
    elif mode == "hybrid":
        constructor = constructor_from_config(config)
        pipeline_server = HybridServer(
            **config["server"],
            pipeline=constructor,
            callbacks=callbacks,
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
        config, args.mode, args.verbose, args.benchmark, ServerCallbacks()
    )
    pipeline_server.start(sample_file)


def setup_log(log_path: Optional[str]):
    if log_path is None:
        return

    log = open(log_path, "w")
    sys.stderr = log
    sys.stdout = log


if __name__ == "__main__":
    args = parse_args()
    log_path: Optional[str]
    if args.silent:
        log_path = os.path.devnull
    elif args.log is None:
        log_path = None
    else:
        log_path = args.log

    setup_log(log_path)
    main(args)
