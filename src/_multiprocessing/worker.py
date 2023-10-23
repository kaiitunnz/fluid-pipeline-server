import multiprocessing
from multiprocessing.connection import Connection
from typing import Any, Callable, Optional


class Worker:
    func: Callable
    pipe: Connection
    name: Optional[str]
    module: Any
    process: Optional[multiprocessing.Process] = None

    def __init__(
        self, func: Callable, pipe: Connection, module: Any, name: Optional[str] = None
    ):
        self.func = func
        self.pipe = pipe
        self.name = name
        self.module = module

    def start(self):
        self.process = multiprocessing.Process(
            target=self.serve, name=self.name, args=(self.module,), daemon=False
        )
        self.process.start()

    def serve(self, module: Any):
        while True:
            print(f"[{self.name}] waiting for a message...")
            pipe, args = self.pipe.recv()
            print(f"[{self.name}] got {args}")
            pipe.send(self.func(*args, module=module))

    def terminate(self, join: bool = True, force: bool = False):
        if self.process is None:
            raise ValueError("The worker process has not been started.")
        if not force:
            while self.pipe.poll():
                pass
        self.process.terminate()
        if join:
            self.process.join()
