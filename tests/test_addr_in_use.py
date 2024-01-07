import copy
import os
from typing import Any, Dict, Optional

import tests.test_utils as tu
from tests.test_utils import run_dummy_server, test_server


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
    new_config["server"]["num_instances"] = 1
    new_config["ui_filter"]["dummy"] = True
    new_config["text_recognizer"]["dummy"] = True
    new_config["icon_labeler"]["dummy"] = True

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
) -> bool:
    dummy_process = run_dummy_server(get_dummy_config(config))
    if dummy_process is None:
        return False

    tu.server_count -= 1
    success = test_server(
        get_test_config(config),
        mode,
        verbose,
        benchmark_file,
        fname,
        json_file,
        chunk_size,
        scale,
        result_dir,
    )

    dummy_process.terminate()
    dummy_process.join()

    return success
