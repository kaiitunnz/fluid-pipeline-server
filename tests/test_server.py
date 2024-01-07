import copy
import os
import unittest
from typing import Any, Dict, Optional

import tests.test_config as cf
import tests.test_utils as tu
from tests import test_addr_in_use, test_basic, test_cuda_oom


class TestPipelineServer(unittest.TestCase):
    config: Dict[str, Any]
    mode: Optional[str] = None

    @classmethod
    def setUpClass(cls):
        if cls.mode is None:
            raise unittest.SkipTest("Invalid server mode.")
        if cf.SERVER_LOG_DIR is not None:
            os.makedirs(cf.SERVER_LOG_DIR, exist_ok=True)
        if cf.TEST_RESULTS_DIR is not None:
            os.makedirs(cf.TEST_RESULTS_DIR, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        os.remove(cf.BENCHMARK_FILE)

    def setUp(self):
        self.config = copy.deepcopy(cf.SERVER_CONFIG)
        self.config["test"] = {}
        self.config["server"]["port"] += tu.server_count

    def get_test_config(self, num_instances: Optional[int] = None) -> Dict[str, Any]:
        new_config = copy.deepcopy(self.config)
        if cf.SERVER_LOG_DIR is not None:
            new_config["test"]["log_path"] = os.path.join(
                cf.SERVER_LOG_DIR, f"{self.mode}.log"
            )
        if num_instances is not None:
            new_config["test"]["num_instances"] = num_instances
        return new_config

    def test_basic(self):
        assert self.mode is not None

        success = test_basic.test(
            self.get_test_config(),
            self.mode,
            cf.VERBOSE,
            cf.BENCHMARK_FILE,
            cf.TEST_IMAGE_FILE,
            cf.TEST_JSON_FILE,
            cf.CHUNK_SIZE,
            cf.SCALE,
            cf.TEST_RESULTS_DIR,
        )
        self.assertTrue(success)

    def test_cuda_oom(self):
        assert self.mode is not None

        success = test_cuda_oom.test(
            self.get_test_config(cf.OOM_NUM_INSTANCES),
            self.mode,
            cf.VERBOSE,
            cf.BENCHMARK_FILE,
            cf.TEST_IMAGE_FILE,
            cf.TEST_JSON_FILE,
            cf.CHUNK_SIZE,
            cf.SCALE,
            cf.TEST_RESULTS_DIR,
        )
        self.assertFalse(success)

    def test_addr_in_use(self):
        assert self.mode is not None

        success = test_addr_in_use.test(
            self.get_test_config(),
            self.mode,
            cf.VERBOSE,
            cf.BENCHMARK_FILE,
            cf.TEST_IMAGE_FILE,
            cf.TEST_JSON_FILE,
            cf.CHUNK_SIZE,
            cf.SCALE,
            cf.TEST_RESULTS_DIR,
        )
        self.assertFalse(success)


class TestHybridServer(TestPipelineServer):
    mode: str = "hybrid"


class TestMultiprocessServer(TestPipelineServer):
    mode: str = "multiprocess"


class TestMultithreadServer(TestPipelineServer):
    mode: str = "multithread"


class TestSequentialServer(TestPipelineServer):
    mode: str = "sequential"
