import json
import multiprocessing as mp
import numpy as np
import os
import sys
from PIL import Image
from ctypes import c_int
from multiprocessing.synchronize import Condition
from typing import Any, Dict, Optional

sys.path.append(os.environ["FLUID_AI_PATH"])

from argparse import ArgumentParser, Namespace
from fluid_ai.utils import plot_ui_elements

from client import print_ui_info, request
from main import init_pipeline_server
from src.utils import parse_results

CONFIG_FILE = "config.json"
BENCHMARK_FILE = os.path.join("test", "tmp.csv")
TEST_IMAGE_FILE = os.path.join("test", "img1.jpg")
TEST_JSON_FILE = os.path.join("test", "img1.json")
TEST_RESULTS_DIR = os.path.join("test", "res")
MODES = ("hybrid", "multiprocess", "multithread", "sequential")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument(
        "-l", "--log", action="store", help="Path to the log file", default=None
    )
    parser.add_argument("-f", "--file", action="store", default=TEST_IMAGE_FILE)
    parser.add_argument("-j", "--json", action="store", default=TEST_JSON_FILE)
    parser.add_argument(
        "-cf",
        "--config-file",
        action="store",
        help="Path to the config file",
        default=CONFIG_FILE,
    )
    parser.add_argument(
        "-bf",
        "--benchmark-file",
        action="store",
        help="Path to the benchmark file",
        default=BENCHMARK_FILE,
    )
    parser.add_argument(
        "--chunk-size",
        action="store",
        default=1024,
        type=int,
        help="Chunk size to read",
    )
    parser.add_argument(
        "--scale",
        action="store",
        default=0.5,
        type=float,
        help="Scale of the result image to display",
    )
    parser.add_argument(
        "--result-dir",
        action="store",
        help="Path to store result files",
        default=TEST_RESULTS_DIR,
    )
    return parser.parse_args()


def on_ready(cond: Condition, is_ready):
    with cond:
        is_ready.value = 1  # type: ignore
        cond.notify()


def on_failure(cond: Condition, is_ready):
    with cond:
        is_ready.value = -1  # type: ignore
        cond.notify()


def run_server(
    mode: str,
    verbose: bool,
    benchmark_file: str,
    config: Dict[str, Any],
    cond: Condition,
    is_ready,
) -> mp.Process:
    def run():
        sample_file = config["server"].pop("sample_file")
        on_failure_ = lambda: on_failure(cond, is_ready)
        server = init_pipeline_server(
            config, mode, verbose, benchmark_file, on_failure_
        )
        server._on_ready = lambda: on_ready(cond, is_ready)
        server._on_failure = on_failure_
        server.start(sample_file)

    process = mp.Process(target=run)
    process.start()
    return process


def test_server(
    config: Dict[str, Any],
    mode: str,
    verbose: bool,
    benchmark_file: str,
    fname: str,
    json_file: str,
    chunk_size: int,
    scale: float,
    result_dir: Optional[str],
) -> bool:
    print(f"Testing '{mode}' server...", flush=True)
    cond = mp.Condition()
    is_ready = mp.Value(c_int, 0)
    process = run_server(mode, verbose, benchmark_file, config, cond, is_ready)

    with cond:
        cond.wait_for(lambda: is_ready.value != 0)  # type: ignore
        success = is_ready.value > 0  # type: ignore

    if success:
        try:
            results = request(
                config["server"]["hostname"],
                config["server"]["port"],
                fname,
                json_file,
                chunk_size,
            )
            img = np.asarray(Image.open(fname))
            elems = parse_results(img, results)
            if result_dir is not None:
                os.makedirs(result_dir, exist_ok=True)
                plot_ui_elements(
                    img, elems, scale, os.path.join(result_dir, f"{mode}.jpg")
                )
                print_ui_info(elems, os.path.join(result_dir, f"{mode}.txt"))
        except Exception:
            success = False

    process.terminate()
    process.join()

    if success:
        print("OK\n", flush=True)
    else:
        print("FAILED\n", flush=True)

    return success


def test(args: Namespace):
    with open(args.config_file, "r") as f:
        config = json.load(f)
    success_cnt = 0
    for mode in MODES:
        if test_server(
            config,
            mode,
            args.verbose,
            args.benchmark_file,
            args.file,
            args.json,
            args.chunk_size,
            args.scale,
            args.result_dir,
        ):
            success_cnt += 1
        config["server"]["port"] += 1
    os.remove(args.benchmark_file)

    print(f"Test finished: {success_cnt} out of {len(MODES)} tests passed.")


if __name__ == "__main__":
    args = parse_args()
    log_path = args.log

    if log_path is None:
        test(args)
    else:
        with open(log_path, "w") as log:
            sys.stderr = log
            sys.stdout = log
            test(args)
