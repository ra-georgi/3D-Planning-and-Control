from abc import ABC, abstractmethod

class Planner(ABC):

    def __init__(self, cfg):
        self.sim_params = cfg

    @abstractmethod
    def calculate_trajectory(self): #-> str:
        """Calculate and return time parameterized Trajectory"""
        


