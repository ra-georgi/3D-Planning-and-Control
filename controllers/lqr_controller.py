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
        self.waypoints  = self.sim_params["world"]["waypoints"]


    def calculate_control(self,state,t): 

        return u
    


    def set_trajectory(self, trajectory):
        # self.trajectory_object = trajectory
        self.lqr_K = self.calculate_gains(trajectory)

    def calculate_gains(self, trajectory_object):
        
        tf = self.waypoints[-1]["t"]
        times = np.arange(0, tf, self.controller_dt)
        num_steps = len(times)
        # num_steps = int((tf/self.controller_dt)) #+ 1

        P = np.zeros((num_steps, 12, 12))
        P[-1,:,:] = self.Qf

        lqr_gains = [] #List of dictionaries
        K = np.zeros((4, 12))
        # K = np.zeros((num_steps-1,  4, 12))   #associate K with a time in a dictionary

        for k in range(N-1,-1,-1):
            # Get required spline values, convert to a full state, linearize about state to get A and B, apply modification for quaternions
            # pos_des, vel_des, acc_des = trajectory_object.evaluate_trajectory(times[k])
            A, B = self.calculate_jacobians()

        N = len(A_seq)
        n, m = B_seq[0].shape[0], B_seq[0].shape[1]
        P = [None]*(N+1)
        K = [None]*N
        P[N] = Qf

        for k in range(N-1, -1, -1):
            A, B = A_seq[k], B_seq[k]
            Q, R = Q_seq[k], R_seq[k]
            S = R + B.T @ P[k+1] @ B
            K[k] = np.linalg.solve(S, B.T @ P[k+1] @ A)
            P[k] = Q + A.T @ P[k+1] @ (A - B @ K[k])       