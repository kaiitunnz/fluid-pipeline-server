from threading import Lock
from typing import Any, Dict, List, Optional, TextIO

Entry = List[Any]


class Benchmarker:
    metrics: List[str]
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
        self.metrics = metrics
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

    def validate_entry(self, entry: Entry):
        return (isinstance(entry, Entry.__origin__)) and (  # type: ignore
            len(entry) == len(self.metrics)
        )

    def get_record_dict(self) -> Dict[str, List[Any]]:
        return {
            metric: list(values)
            for metric, values in zip(self.metrics, zip(*self.record))
        }

    def __del__(self):
        self.output_file.close()

    def _write_entry(self, entry: Entry):
        if self.lock is None:
            self.output_file.write(",".join(str(v) for v in entry) + "\n")
            self.output_file.flush()
        else:
            self.lock.acquire()
            self.output_file.write(",".join(str(v) for v in entry) + "\n")
            self.output_file.flush()
            self.lock.release()
