from controllers.controller import Controller
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import block_diag

class LQR_Controller(Controller):

    def __init__(self, cfg):
        super().__init__(cfg)
        with open("config/controllers/lqr.yaml", "r") as f:
            self.controller_params = yaml.safe_load(f)

        self.sim_dt = self.sim_params["time"]["dt"]
        self.controller_dt   = 1/self.controller_params["rates_hz"]
        if (self.sim_dt > self.controller_dt):
              print("Warning: Simulation time step greater than LQR time step")

        self.Q  = self.controller_params["state_cost_multiplier"]*block_diag( np.eye(6), self.controller_params["orientation_cost_multiplier"]*np.eye(6) )
        self.R  = self.controller_params["control_cost_multiplier"]*np.diag(np.ones(4))
        self.Qf = self.controller_params["terminal_state_cost_multiplier"]*block_diag( np.eye(6), self.controller_params["orientation_cost_multiplier"]*np.eye(6) )

        self.controller_name = "LQR"    


    def calculate_control(self,state,t): 

        return u
    


    def set_trajectory(self, trajectory):
        self.trajectory_object = trajectory