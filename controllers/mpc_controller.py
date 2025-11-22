from controllers.controller import Controller
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import block_diag
import casadi

class MPC_Controller(Controller):

    def __init__(self, cfg):
        super().__init__(cfg)
        with open("config/controllers/mpc.yaml", "r") as f:
            self.controller_params = yaml.safe_load(f)

        self.sim_dt          = self.sim_params["time"]["dt"]
        self.controller_dt   = 1/self.controller_params["rates_hz"]
        if (self.sim_dt > self.controller_dt):
              print("Warning: Simulation time step greater than MPC time step")

        self.Q  = self.controller_params["state_cost_multiplier"]*block_diag( np.eye(6), self.controller_params["orientation_cost_multiplier"]*np.eye(6) )
        self.R  = self.controller_params["control_cost_multiplier"]*np.diag(np.ones(4))
        self.Qf = self.controller_params["terminal_state_cost_multiplier"]*block_diag( np.eye(6), self.controller_params["orientation_cost_multiplier"]*np.eye(6) )

        self.controller_name = "MPC"    
        self.waypoints  = self.sim_params["world"]["waypoints"]

        self.mass =       self.sim_params["quadcopter"]["mass"]
        self.I_xx =       self.sim_params["quadcopter"]["I_xx"]
        self.I_yy =       self.sim_params["quadcopter"]["I_yy"]
        self.I_zz =       self.sim_params["quadcopter"]["I_zz"]
        self.arm_length = self.sim_params["quadcopter"]["arm_length"]
        self.kf =         self.sim_params["quadcopter"]["motor"]["kf"]
        self.km =         self.sim_params["quadcopter"]["motor"]["km"]
        self.g =          self.sim_params["constants"]["acc_gravity"]
        self.dt =         self.sim_params["time"]["dt"]

        # self.pad_matrix = jnp.block([
        #                 [jnp.zeros([1,3])],
        #                 [jnp.eye(3)]
        #                 ])  

        # self.motor_matrix = np.array([
        #         [self.kf,                     self.kf,                 self.kf,                  self.kf],
        #         [0,                           self.arm_length*self.kf, 0,                        -self.arm_length*self.kf],
        #         [-self.arm_length*self.kf,    0,                       self.arm_length*self.kf,       0],
        #         [self.km,                     -self.km,                self.km,                  -self.km]
        # ]) 

        # self.final_K = None

    def set_trajectory(self, trajectory):
        self.trajectory_object = trajectory


    def calculate_control(self,state,t): 

        return u