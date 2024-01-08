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
    full: Optional[bool] = None
    default_test_config: Dict[str, Any] = {
        "server_timeout": cf.SERVER_TIMEOUT,
        "socket_timeout": cf.SOCKET_TIMEOUT,
    }

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

    def get_test_config(self, **kwargs) -> Dict[str, Any]:
        assert self.mode is not None

        new_config = copy.deepcopy(self.config)
        default_test_config = copy.deepcopy(self.default_test_config)

        if cf.SERVER_LOG_DIR is not None:
            if self.full is None:
                fname = self.mode
            elif self.full:
                fname = f"full_{self.mode}"
            else:
                fname = f"lite_{self.mode}"
            new_config["test"]["log_path"] = os.path.join(
                cf.SERVER_LOG_DIR, f"{fname}.log"
            )
            new_config["test"]["result_fname"] = fname

        new_config["test"].update(default_test_config | kwargs)
        return new_config

    def test_basic(self):
        assert self.mode is not None

        result = test_basic.test(
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
        self.assertTrue(result.assert_true(), result.error)

    def test_cuda_oom(self):
        assert self.mode is not None

        result = test_cuda_oom.test(
            self.get_test_config(memory_fraction=cf.OOM_MEMORY_FRACTION),
            self.mode,
            cf.VERBOSE,
            cf.BENCHMARK_FILE,
            cf.TEST_IMAGE_FILE,
            cf.TEST_JSON_FILE,
            cf.CHUNK_SIZE,
            cf.SCALE,
            cf.TEST_RESULTS_DIR,
        )
        self.assertFalse(result.assert_false(), result.error)

    def test_addr_in_use(self):
        assert self.mode is not None

        result = test_addr_in_use.test(
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
        self.assertFalse(result.assert_false(), result.error)


class TestFullServer(TestPipelineServer):
    full: Optional[bool] = True

    def get_test_config(self, **kwargs) -> Dict[str, Any]:
        config = super().get_test_config(**kwargs)
        config["ui_filter"]["dummy"] = False
        config["text_recognizer"]["dummy"] = False
        config["icon_labeler"]["dummy"] = False
        return config


class TestLiteServer(TestPipelineServer):
    full: Optional[bool] = False

    def get_test_config(self, **kwargs) -> Dict[str, Any]:
        config = super().get_test_config(**kwargs)
        config["ui_filter"]["dummy"] = False
        config["text_recognizer"]["dummy"] = True
        config["icon_labeler"]["dummy"] = True
        return config


class TestFullHybridServer(TestFullServer):
    mode: str = "hybrid"


class TestFullMultiprocessServer(TestFullServer):
    mode: str = "multiprocess"


class TestFullMultithreadServer(TestFullServer):
    mode: str = "multithread"


class TestFullSequentialServer(TestFullServer):
    mode: str = "sequential"


class TestLiteHybridServer(TestLiteServer):
    mode: str = "hybrid"


class TestLiteMultiprocessServer(TestLiteServer):
    mode: str = "multiprocess"


class TestLiteMultithreadServer(TestLiteServer):
    mode: str = "multithread"


class TestLiteSequentialServer(TestLiteServer):
    mode: str = "sequential"
