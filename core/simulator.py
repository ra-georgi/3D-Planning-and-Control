
class Simulator():

    def __init__(self, cfg):
        self.params = cfg

    def simulate(self): #-> str:
        """Simulate Quadcopter Flight"""
        dt = self.params["time"]["dt"]
        tf = self.params["time"]["duration"]
        
    def take_step(self):
        """ Simulate for a single time step"""

    def rk4_step(self):
        """" Lol"""

