import json
import os
import sys
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch

sys.path.append(os.path.abspath(".."))

from fluid_ai.icon import ClassifierIconLabeller

# from fluid_ai.ocr import EasyOCR
from fluid_ai.ocr import DummyOCR
from fluid_ai.pipeline import UiDetectionPipeline
from fluid_ai.ui.detection import YoloUiDetector

from src.threading.server import PipelineServer as ThreadingServer
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
        choices=["sequential", "threading"],
        default="threading",
        help="Pipeline server mode",
    )
    return parser.parse_args()


def pipeline_from_config(config: Dict[str, Any]) -> UiDetectionPipeline:
    detector = YoloUiDetector(
        config["ui_detector"]["path"],
        device=torch.device(config["ui_detector"]["device"]),
    )
    # text_recognizer = EasyOCR(batch_size=config["text_recognizer"]["batch_size"])
    text_recognizer = DummyOCR()
    icon_labeller = ClassifierIconLabeller(
        config["icon_labeller"]["path"],
        batched=True,
        device=torch.device(config["icon_labeller"]["device"]),
    )
    return UiDetectionPipeline(
        detector,
        text_recognizer,
        config["special_elements"]["text"],
        icon_labeller,
        config["special_elements"]["icon"],
    )


def main(args: Namespace):
    with open("config.json", "r") as f:
        config = json.load(f)
    sample_file = config["server"].pop("sample_file")
    pipeline = pipeline_from_config(config)
    if args.mode == "sequential":
        pipeline_server = SequentialServer(
            **config["server"],
            pipeline=pipeline,
            verbose=args.verbose,
            benchmark=args.benchmark is not None,
            benchmark_file=args.benchmark,
        )
    elif args.mode == "threading":
        pipeline_server = ThreadingServer(
            **config["server"],
            pipeline=pipeline,
            verbose=args.verbose,
            benchmark=args.benchmark is not None,
            benchmark_file=args.benchmark,
        )
    else:
        raise NotImplementedError()
    pipeline_server.warmup(sample_file)
    pipeline_server.start()


if __name__ == "__main__":
    args = parse_args()
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
