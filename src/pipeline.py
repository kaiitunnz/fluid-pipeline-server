from abc import abstractmethod


class PipelineServerInterface:
    @abstractmethod
    def start(self, _):
        raise NotImplementedError()
