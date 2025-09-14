from abc import ABC, abstractmethod


class AbstractModel(ABC):
    """
    Neural network model interface. All custom neural network model classes must implement this interface.
    """
    def __init__(self, input_size, output_size, word_size):
        self.input_size = input_size
        self.output_size = output_size
        self.word_size = word_size

    @abstractmethod
    def make_net(self):
        pass
