from abc import ABC, abstractmethod


class Action(ABC):
    @abstractmethod
    def perform(self, *args, **kwargs):
        """
        Abstract function to be implemented by the child classes
        :return:
        """