from controllers.controller import Controller
import numpy as np

class Hover_Controller(Controller):

    def __init__(self, cfg):
        super().__init__(cfg)

    def calculate_control(self,x,t): #-> str:
        """Calculate and return control input"""
        m = self.params["quadcopter"]["mass"]
        g = self.params["constants"]["acc_gravity"]
        u = (m*g)/4
        return u*np.ones((4, 1))
        


