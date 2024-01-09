import logging
import os
import signal
import socket as sock
import sys
from abc import abstractmethod
from typing import Any, Callable, Optional

from src.logger import ILogger

Callback = Callable[[], Any]


class ServerCallbacks:
    """
    Server callbacks.

    Attributes
    ----------
    on_start : Callback
        Callback to be called when the pipeline server is starting up.
    on_ready : Callback
        Callback to be called when the pipeline server is ready to serve.
    on_failure : Callback
        Callback to be called when the pipeline server fails to start.
    on_exit : Callback
        Callback to be called when the pipeline server exits.
    """

    on_start: Callback
    on_ready: Callback
    on_failure: Callback
    on_exit: Callback

    def __init__(
        self,
        on_start: Optional[Callback] = None,
        on_ready: Optional[Callback] = None,
        on_failure: Optional[Callback] = None,
        on_exit: Optional[Callback] = None,
    ):
        """
        Parameters
        ----------
        on_start : Optional[Callback]
            Callback to be called when the pipeline server is starting up.
        on_ready : Optional[Callback]
            Callback to be called when the pipeline server is ready to serve.
        on_failure : Optional[Callback]
            Callback to be called when the pipeline server fails to start.
        on_exit : Optional[Callback]
            Callback to be called when the pipeline server exits.
        """
        self.on_start = on_start or self.__class__.default
        self.on_ready = on_ready or self.__class__.default
        self.on_failure = on_failure or self.__class__.default
        self.on_exit = on_exit or self.__class__.default

    @staticmethod
    def default():
        """Default callback

        Do nothing.
        """
        pass


class IPipelineServer:
    """
    An interface of a UI detection pipeline server.

    Attributes
    ----------
    hostname : str
        Host name.
    port : int
        Port to listen to client connections.
    socket : Optional[sock.socket]
        Server socket.
    logger : ILogger
        Logger to log the UI detection process.
    """

    hostname: str
    port: int
    socket: Optional[sock.socket]
    logger: ILogger

    _callbacks: ServerCallbacks
    _is_ready: bool
    _pid: int

    def __init__(
        self,
        hostname: str,
        port: int,
        socket: Optional[sock.socket],
        logger: ILogger,
        callbacks: ServerCallbacks,
    ):
        """
        Parameters
        ----------
        hostname : str
            Host name.
        port : int
            Port to listen to client connections.
        socket : Optional[sock.socket]
            Server socket.
        logger : ILogger
            Logger to log the UI detection process.
        callbacks : ServerCallbacks
            Server callbacks.
        """
        self.hostname = hostname
        self.port = port
        self.socket = socket
        self.logger = logger
        self._callbacks = callbacks

        self._is_ready = False
        self._pid = os.getpid()

    def getpid(self) -> int:
        """Gets the ID of the pipeline server's main process

        Returns
        -------
        int
            Process ID of the pipeline server
        """
        return self._pid

    @abstractmethod
    def serve(self, _):
        """Serves the UI detection pipeline server"""
        raise NotImplementedError()

    def start(self, arg: Any):
        """Starts the UI detection pipeline server

        Parameters
        ----------
        arg : Any
            Argument to be passed to the `IPipelineServer.serve()` method.
        """
        self._callbacks.on_start()
        self.serve(arg)

    def bind(self) -> Optional[sock.socket]:
        """Binds a socket to the server's hostname and port

        Other components of the server must be ready to serve prior to calling this
        function.

        Returns
        -------
        Optional[socket]
            Server socket.
        """
        socket = None
        try:
            socket = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
            socket.bind((self.hostname, self.port))
            socket.listen(1)
            self.logger.info(
                f'Pipeline server started serving at "{self.hostname}:{self.port} (PID={os.getpid()})".'
            )
            self._is_ready = True
            self._callbacks.on_ready()
            return socket
        except OSError as e:
            self.logger.error(f"Fatal error occurred: {e}")
            if socket is not None:
                socket.close()
            os.kill(self._pid, signal.SIGTERM)
            return None

    def exit(self, code: int):
        """Terminates the pipeline server process

        Parameters
        ----------
        code : int
            Exit code.
        """
        if self._is_ready:
            self._callbacks.on_exit()
        else:
            self._callbacks.on_failure()
        sys.exit(code)

    @classmethod
    def _init_logger(cls, verbose: bool = True) -> logging.Logger:
        """Initializes the logger to log server events

        Parameters
        ----------
        verbose : bool
            Whether to log server events verbosely.

        Returns
        -------
        Logger
            Initialized logger.
        """
        fmt = "[%(asctime)s | %(name)s] [%(levelname)s] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        logging.basicConfig(format=fmt, datefmt=datefmt)
        logger = logging.getLogger(cls.__name__)
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        return logger
