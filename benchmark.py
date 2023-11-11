import json
import multiprocessing as mp
import os
import socket
import sys
import time
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import pandas as pd  # type: ignore

sys.path.append(os.path.abspath(".."))

from src.utils import readall

NUM_REQUESTS = 100
NUM_CONCURRENT = mp.cpu_count()
CONFIG_FILE = "config.json"
SAMPLE_IMAGE = os.path.join("res", "sample.jpg")
RESULT_PATH = "benchmark.csv"

MAX_ERR_COUNT = 10

results = None
sample_image = None


def init(results_: Any, sample_image_: str):
    global results, sample_image
    results = results_
    sample_image = sample_image_


def load_server_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["server"]


def request(i: int):
    config = load_server_config(CONFIG_FILE)
    chunk_size = config["chunk_size"]
    err_count = 0

    s = None
    assert not (results is None or sample_image is None)
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((config["hostname"], config["port"]))

            start = time.time()

            n = os.path.getsize(sample_image)
            s.sendall(n.to_bytes(4, "big", signed=False))
            with open(sample_image, "rb") as f:
                s.sendfile(f)

            n = int.from_bytes(readall(s, 4, chunk_size), "big", signed=False)
            data = readall(s, n, chunk_size)

            elapsed = time.time() - start
            break
        except Exception as e:
            assert s is not None
            err_count += 1
            print(
                f"[Request {i} on Port {s.getsockname()[1]}] Failed with exception {e}.",
                end=" ",
                file=sys.stderr,
            )
            s.close()
            if err_count > MAX_ERR_COUNT:
                print("Terminating...", file=sys.stderr)
                sys.exit(1)
            else:
                print("Retrying...", file=sys.stderr)

    metrics = json.loads(data).get("metrics", {"keys": [], "values": []})
    metrics["keys"].append("Response time")
    metrics["values"].append(elapsed)

    results.append(metrics)


def compute_metrics(metrics: pd.DataFrame) -> pd.Series:
    average = metrics.mean(axis=0)
    average.index = pd.Index(["average " + idx.lower() for idx in metrics.columns])
    stdev = metrics.std(axis=0)
    stdev.index = pd.Index(["stdev " + idx.lower() for idx in metrics.columns])
    summary = pd.concat([average, stdev])
    summary = summary.loc[
        [x for avg_stdev in zip(average.index, stdev.index) for x in avg_stdev]
    ]
    summary["throughput"] = 1 / summary["average response time"]
    summary["peak response time"] = metrics["Response time"].max()
    return summary


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-n", action="store", type=int, help="Number of requests", default=NUM_REQUESTS
    )
    parser.add_argument(
        "-c",
        action="store",
        type=int,
        help="Number of concurrent requests",
        default=NUM_CONCURRENT,
    )
    parser.add_argument(
        "-f",
        action="store",
        help="Path to the sample image file",
        default=SAMPLE_IMAGE,
    )
    parser.add_argument(
        "-r",
        action="store",
        help="Path to the result file to be stored",
        default=RESULT_PATH,
    )
    parser.add_argument(
        "--config", action="store", help="Path to the config file", default=CONFIG_FILE
    )
    return parser.parse_args()


def main(args: Namespace):
    print(
        f"Testing the server with {args.n} requests ({args.c} concurrent requests)..."
    )
    print(f'Configuration file: "{args.config}"')
    print(f'Sample file: "{args.f}" ({os.path.getsize(args.f)} bytes)')

    with mp.Manager() as manager:
        r = manager.list()
        with mp.Pool(processes=args.c, initializer=init, initargs=(r, args.f)) as pool:
            pool.map(request, range(args.n))
        bench_results = pd.DataFrame(
            [list(row["values"]) for row in r], columns=r[0]["keys"]
        )
        bench_results.to_csv(args.r)
        metrics = compute_metrics(bench_results)
        print("Computed metrics:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
