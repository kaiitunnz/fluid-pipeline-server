import json
import multiprocessing as mp
import os
from PIL import Image
from ctypes import c_int
from multiprocessing.synchronize import Condition
from typing import Any, Dict, Optional

import numpy as np
from fluid_ai.utils import plot_ui_elements

from client import print_ui_info, request
from main import init_pipeline_server, setup_log
from src.utils import parse_results

server_count: int = 0


def on_ready(cond: Condition, is_ready):
    with cond:
        is_ready.value = 1  # type: ignore
        cond.notify()


def on_failure(cond: Condition, is_ready):
    with cond:
        is_ready.value = -1  # type: ignore
        cond.notify()


def run_dummy_server(
    config: Dict[str, Any],
    mode: str = "hybrid",
    verbose: bool = False,
    benchmark_file: Optional[str] = None,
) -> Optional[mp.Process]:
    cond = mp.Condition()
    is_ready = mp.Value(c_int, 0)
    process = run_server(mode, verbose, benchmark_file, config, cond, is_ready)

    with cond:
        cond.wait_for(lambda: is_ready.value != 0)  # type: ignore
        success = is_ready.value > 0  # type: ignore

    if success:
        return process
    return None


def run_server(
    mode: str,
    verbose: bool,
    benchmark_file: Optional[str],
    config: Dict[str, Any],
    cond: Condition,
    is_ready,
) -> mp.Process:
    global server_count

    server_count += 1

    def run():
        sample_file = config["server"].pop("sample_file")
        log_path = config["test"].get("log_path")
        setup_log(log_path)
        print("Running a server with the following config:")
        print(json.dumps(config, indent=4), flush=True)
        print()
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
                fname = config["test"]["result_fname"]
                plot_ui_elements(
                    img, elems, scale, os.path.join(result_dir, f"{fname}.jpg")
                )
                print_ui_info(elems, os.path.join(result_dir, f"{fname}.txt"))
        except Exception:
            success = False

    process.terminate()
    process.join()

    return success
