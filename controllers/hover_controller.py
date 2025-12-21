from controllers.controller import Controller
import numpy as np
import yaml

class Hover_Controller(Controller):

    def __init__(self, cfg):
        super().__init__(cfg)
        with open("config/controllers/hover.yaml", "r") as f:
            self.controller_params = yaml.safe_load(f)
        self.controller_dt = self.controller_params["rates_hz"]
        self.controller_name = "Hover Controller"

    def set_trajectory(self, trajectory):
        pass
        # self.trajectory_object = trajectory
        # self.lqr_K = self.calculate_gains(trajectory)

    def calculate_control(self,x,t): #-> str:
        """Calculate and return control input"""
        m  = self.sim_params["quadcopter"]["mass"]
        g  = self.sim_params["constants"]["acc_gravity"]
        kf = self.sim_params["quadcopter"]["motor"]["kf"]
        u = (1/kf)*(m*g)/4
        u_array = u*np.ones((4))
        # u_array[2] = 1.01*u_array[3]
        # u_array[3] = 1.01*u_array[3]
    
        return u_array
        


