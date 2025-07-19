from abc import ABC, abstractmethod


class AbstractModel(ABC):
    """
    神经网络模型接口，所有的自定义神经网络模型类都要实现这个接口
    """
    def __init__(self, input_size, output_size, word_size):
        self.input_size = input_size
        self.output_size = output_size
        self.word_size = word_size

    @abstractmethod
    def make_net(self):
        pass
