import copy
import os
import unittest
from multiprocessing import Process, Semaphore, SimpleQueue
from typing import Any, Callable, Dict, Optional, Tuple

from src.server import ServerCallbacks
from tests.test_utils import TestResult, test_server

import torch


def occupy_gpu_memory(channel: SimpleQueue, fraction: float):
    tensors = []
    try:
        if not torch.cuda.is_available():
            raise Exception("CUDA is not available.")

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
    Semaphore(0).acquire()


def get_on_ready_callback(
    memory_fraction: float,
) -> Tuple[Process, Callable[[bool], Any]]:
    channel: SimpleQueue[Optional[Exception]] = SimpleQueue()
    dummy_process = Process(target=occupy_gpu_memory, args=(channel, memory_fraction))
    return dummy_process, (
        lambda success: run_dummy_process(success, channel, dummy_process)
    )


def run_dummy_process(success: bool, channel: SimpleQueue, process: Process):
    if not success:
        return
    process.start()
    e = channel.get()
    if e is not None:
        process.kill()
        process.join()
        raise unittest.SkipTest(f"Fail to initialize the test: {e}")


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
    dummy_process, on_ready = get_on_ready_callback(config["test"]["memory_fraction"])

    try:
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
            ServerCallbacks(),
            warm_up=False,
            on_ready=on_ready,
        )
    except ConnectionError:
        result = TestResult(False, None)

    if dummy_process.is_alive():
        dummy_process.kill()
        dummy_process.join()

    return result
