from abc import ABC, abstractmethod

class Algorithm(ABC):
    @abstractmethod
    def get_count(self):
        pass

    @abstractmethod
    def get_wavefront(self, meas_count):
        pass

    @abstractmethod
    def post_process(self, feedback_set):
        pass
