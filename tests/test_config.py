import json
import os

BENCHMARK_FILE = os.path.join("tests", "tmp.csv")
CONFIG_FILE = "config.json"
SERVER_LOG_DIR = os.path.join("tests", "log")
TEST_IMAGE_FILE = os.path.join("tests", "img1.jpg")
TEST_JSON_FILE = os.path.join("tests", "img1.json")
TEST_RESULTS_DIR = os.path.join("tests", "res")

VERBOSE = True
CHUNK_SIZE = 1024
SCALE = 0.5
OOM_MEMORY_FRACTION = 0.99
SERVER_TIMEOUT = 20.0
SOCKET_TIMEOUT = 20.0

with open(CONFIG_FILE, "r") as f:
    SERVER_CONFIG = json.load(f)
