from controllers.controller import Controller
import yaml
import numpy as np

class Cascade_PID(Controller):

    def __init__(self, cfg):
        super().__init__(cfg)
        with open("config/controllers/cascade_pid.yaml", "r") as f:
            self.controller_params = yaml.safe_load(f)

        self.x_integral     = 0      
        self.y_integral     = 0                          
        self.z_integral     = 0
        self.phi_integral   = 0
        self.theta_integral = 0
        self.psi_integral   = 0

        self.PID_Gains       = self.controller_params["gains"]
        self.outer_loop_dt   = 1/self.controller_params["rates_hz"]["outer_loop"]
        self.inner_loop_dt   = 1/self.controller_params["rates_hz"]["inner_loop"]
        self.outer_loop_time = 0
        self.inner_loop_time = 0


 

    def set_trajectory(self, trajectory):
        self.trajectory_object = trajectory

    def calculate_control(self,x,t): #-> str:
        pos_des, vel_des, acc_des = self.trajectory_object.evaluate_trajectory(t)


        m = self.sim_params["quadcopter"]["mass"]
        g = self.sim_params["constants"]["acc_gravity"]
        u = (m*g)/4
        return u*np.ones((4))
    
    def position_controller(self, pos, pos_des):
        x,x_des = state[0], self.des_state[0,i]
        vx, vx_des = state[7], self.des_state[6,i]
        y,y_des = state[1], self.des_state[1,i]
        vy, vy_des = state[8], self.des_state[7,i]

        self.x_integral += self.h*(x_des-x)
        self.y_integral += self.h*(y-y_des)

        self.t_outer += self.h

        if self.t_outer >= self.outer_loop_h:
                phi_des= np.dot(self.PID_Gains[1,:],[y-y_des, self.y_integral, vy-vy_des])
                if phi_des > 20:
                        phi_des = 20
                elif phi_des < -20:
                        phi_des = -20
                self.des_state[3] = phi_des
                theta_des = np.dot(self.PID_Gains[0,:],[x_des-x, self.x_integral, vx_des-vx])
                if theta_des > 20:
                        theta_des = 20
                elif theta_des < -20:
                        theta_des = -20
                self.des_state[4] = theta_des

                self.t_outer = 0

        u = self.PID_attitude_controller(state,t,i)
        return u
  

    def attitude_controller():
        pass