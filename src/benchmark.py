from abc import abstractmethod
from threading import Lock
from typing import Any, Dict, List, Optional, TextIO

Entry = List[Any]
"""
Benchmark entry.
"""

BENCHMARK_METRICS = [
    "Waiting time",
    "UI detection time",
    "Invalid UI detection time",
    "UI matching time",
    "UI processing time",
    "Text recognition time",
    "Icon labeling time",
    "Processing time",
]
"""
Benchmark metrics.
"""


class IBenchmarker:
    """
    An interface for a benchmarker.

    Attributes
    ----------
    metrics: List[str]
        List of benchmark metric names.
    """

    metrics: List[str]

    def __init__(self, metrics: List[str]):
        self.metrics = metrics

    @abstractmethod
    def add(self, entry: Entry):
        """Adds a benchmark entry to the record

        Parameters
        ----------
        entry : Entry
            Benchmark entry to be added. It must have the scheme specified in `metrics`.
        """
        raise NotImplementedError()


class Benchmarker(IBenchmarker):
    """
    A class for benchmarking a system.

    Attributes
    ----------
    metrics: List[str]
        List of benchmark metric names.
    output_file: TextIO
        File to log the benchmark process to.
    max_num_entries: int
        Maximum number of benchmark entries.
    record: List[Entry]
        Record of benchmark entries.
    lock: Optional[Lock]
        Lock to synchronize access to the output file. `None` if there is no need
        to synchronize.
    """

    output_file: TextIO
    max_num_entries: int
    record: List[Entry]
    lock: Optional[Lock]

    def __init__(
        self,
        metrics: List[str],
        output_file: str,
        header: bool = True,
        max_num_entries: int = 200,
        lock: bool = True,
    ):
        """
        Parameters
        ----------
        metrics : List[str]
            List of benchmark metric names.
        output_file : str
            Path to the file to log the benchmark process to.
        header : bool
            Whether to log the benchmark header.
        max_num_entries: int
            Maximum number of benchmark entries.
        lock : bool
            Whether to use a lock for synchronized logging.
        """
        super().__init__(metrics)

        self.output_file = open(output_file, "w")
        self.record = []
        self.max_num_entries = max_num_entries
        self.lock = Lock() if lock else None

        # Write the header.
        if header:
            self.output_file.write(",".join(metrics) + "\n")
            self.output_file.flush()

    def add(self, entry: Entry):
        if not self.validate_entry(entry):
            raise ValueError("Invalid benchmark entry.")
        if len(self.record) < self.max_num_entries:
            self.record.append(entry)
        self._write_entry(entry)

    def validate_entry(self, entry: Entry) -> bool:
        """Validates the benchmark entry

        Parameters
        ----------
        entry : Entry
            Benchmark entry to be validated.

        Returns
        -------
        bool
            Whether the benchmark entry is valid.
        """
        return (isinstance(entry, Entry.__origin__)) and (  # type: ignore
            len(entry) == len(self.metrics)
        )

    def get_record_dict(self) -> Dict[str, List[Any]]:
        """Gets the benchmark record as a dictionary

        Returns
        -------
        Dict[str, List[Any]]
            Benchmark record dictionary with metric names and values as the keys
            and values, respectively.
        """
        return {
            metric: list(values)
            for metric, values in zip(self.metrics, zip(*self.record))
        }

    def __del__(self):
        self.output_file.close()

    def _write_entry(self, entry: Entry):
        """Writes the benchmark entry to the output file

        Returns
        -------
        entry : Entry
            Entry to be written to the output file.
        """
        if self.lock is None:
            self.output_file.write(",".join(str(v) for v in entry) + "\n")
            self.output_file.flush()
        else:
            self.lock.acquire()
            self.output_file.write(",".join(str(v) for v in entry) + "\n")
            self.output_file.flush()
            self.lock.release()
