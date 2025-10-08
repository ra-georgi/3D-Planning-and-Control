from abc import ABC, abstractmethod

class Controller(ABC):

    def __init__(self, cfg):
        self.sim_params = cfg

    @abstractmethod
    def calculate_control(self): #-> str:
        """Calculate and return control input"""
        


