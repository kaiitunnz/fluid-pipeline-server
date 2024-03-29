import copy
import functools
import multiprocessing as mp
import os
import unittest
from multiprocessing.synchronize import Semaphore
from typing import Any, Dict, Optional

from src.server import ServerCallbacks
from tests.test_utils import TestResult, test_server, on_failure

import torch


def on_start(start_sema: Semaphore, continue_sema: Semaphore):
    start_sema.release()
    continue_sema.acquire()


def occupy_gpu_memory(
    channel: mp.SimpleQueue,
    start_sema: Semaphore,
    continue_sema: Semaphore,
    fraction: float,
):
    if not torch.cuda.is_available():
        channel.put(Exception("CUDA is not available."))
        return
    channel.put(None)

    start_sema.acquire()
    tensors = []
    try:
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            tensors.append(
                torch.empty(
                    int(free * fraction), dtype=torch.int8, device=torch.device(i)
                )
            )
        channel.put(None)
    except Exception as e:
        channel.put(e)
    continue_sema.release()
    mp.Semaphore(0).acquire()


def run_dummy_process(
    channel: mp.SimpleQueue,
    start_sema: Semaphore,
    continue_sema: Semaphore,
    memory_fraction: float,
) -> mp.Process:
    dummy_process = mp.Process(
        target=occupy_gpu_memory,
        args=(channel, start_sema, continue_sema, memory_fraction),
    )
    dummy_process.start()
    e = channel.get()
    if e is not None:
        dummy_process.kill()
        dummy_process.join()
        raise unittest.SkipTest(f"Fail to initialize the test: {e}")
    return dummy_process


def get_test_config(config: Dict[str, Any]) -> Dict[str, Any]:
    new_config = copy.deepcopy(config)

    test_config = new_config.get("test")
    assert isinstance(test_config, dict)
    log_path = test_config.get("log_path")
    if log_path is not None:
        dirname = os.path.dirname(log_path)
        basename = os.path.basename(log_path)
        prefix = __name__.split(".")[-1]
        test_config["log_path"] = os.path.join(dirname, f"{prefix}_{basename}")
    return new_config


def get_dummy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    new_config = copy.deepcopy(config)

    # Set log path.
    test_config = new_config.get("test")
    assert isinstance(test_config, dict)
    test_config["log_path"] = os.devnull

    # Set server config.
    new_config["server"]["num_workers"] = 1
    new_config["server"]["num_instances"] = test_config["num_instances"]

    return new_config


def test(
    config: Dict[str, Any],
    mode: str,
    verbose: bool,
    benchmark_file: str,
    fname: str,
    json_file: str,
    chunk_size: int,
    scale: float,
    result_dir: Optional[str],
) -> TestResult:
    channel: mp.SimpleQueue[Optional[Exception]] = mp.SimpleQueue()
    start_sema = mp.Semaphore(0)
    continue_sema = mp.Semaphore(0)

    dummy_process = run_dummy_process(
        channel, start_sema, continue_sema, config["test"]["memory_fraction"]
    )

    result = test_server(
        get_test_config(config),
        mode,
        verbose,
        benchmark_file,
        fname,
        json_file,
        chunk_size,
        scale,
        result_dir,
        ServerCallbacks(
            on_start=functools.partial(on_start, start_sema, continue_sema),
        ),
        on_server_exit=on_failure,
    )

    start_error = channel.get()
    if start_error is not None:
        result = TestResult(False, start_error)

    if dummy_process.is_alive():
        dummy_process.kill()
        dummy_process.join()

    return result
