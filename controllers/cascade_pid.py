from controllers.controller import Controller
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

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

        self.sim_dt = self.sim_params["time"]["dt"]
        if (self.sim_dt > self.outer_loop_dt) or (self.sim_dt > self.inner_loop_dt):
              print("Warning: Simulation time step greater than PID inner/outer loop time step")

 

    def set_trajectory(self, trajectory):
        self.trajectory_object = trajectory

    def calculate_control(self,x,t): #-> str:
        pos_des, vel_des, acc_des = self.trajectory_object.evaluate_trajectory(t)
        u = self.position_controller(x, pos_des, vel_des, t)

        # m = self.sim_params["quadcopter"]["mass"]
        # g = self.sim_params["constants"]["acc_gravity"]
        # u = (m*g)/4
        # return u*np.ones((4))
    
    def position_controller(self, x, pos_des, vel_des, t):

        x, y, z    = x[0:3]
        vx, vy, vz = x[7:10]
        x_des, y_des, z_des    = pos_des
        vx_des, vy_des, vz_des = vel_des
        
        x_gains = [self.PID_Gains["outer_loop"]["proportional"]["x"], self.PID_Gains["outer_loop"]["integral"]["x"], self.PID_Gains["outer_loop"]["derivative"]["x"]]
        y_gains = [self.PID_Gains["outer_loop"]["proportional"]["y"], self.PID_Gains["outer_loop"]["integral"]["y"], self.PID_Gains["outer_loop"]["derivative"]["y"]]

        if t >= self.outer_loop_time:
            self.outer_loop_time += self.outer_loop_dt

            self.x_integral += self.sim_dt*(x_des-x)
            self.y_integral += self.sim_dt*(y-y_des)
            phi_des= np.dot(y_gains,[y-y_des, self.y_integral, vy-vy_des])
            if phi_des > 20:
                    phi_des = 20
            elif phi_des < -20:
                    phi_des = -20

            theta_des = np.dot(x_gains,[x_des-x, self.x_integral, vx_des-vx])
            if theta_des > 20:
                    theta_des = 20
            elif theta_des < -20:
                    theta_des = -20

            self.t_outer = 0

        u = self.attitude_controller(x, phi_des, theta_des, t)
        return u
  

    def attitude_controller(self, x, phi_des, theta_des, t):
  
        quaternion = R.from_quat(np.array(x[3:7]),scalar_first=True)
        eul = quaternion.as_euler('zyx', degrees=True)

        phi, theta, psi = [eul[2],eul[1],eul[0]]

        psi_des = 0 #TODO: reexamine this later
        ang_vel_x_des = ang_vel_y_des = ang_vel_z_des = 0   #TODO: reexamine this later
        # ang_vel_x, ang_vel_x_des = state[10], self.des_state[9,i]    #using omega from state is an approximation

        ang_vel_x, ang_vel_y, ang_vel_z = x[10:]

    
        z, z_des = state[2], self.des_state[2,i]
        vz, vz_des = state[9], self.des_state[8,i]

        self.z_integral += self.h*(z_des-z)
        self.phi_integral += self.h*(phi_des-phi)
        self.theta_integral += self.h*(theta_des-theta)
        self.psi_integral += self.h*(psi_des-psi)

        T   = np.dot(self.PID_Gains[2,:],[z_des-z, self.z_integral, vz_des-vz])
        M_x = np.dot(self.PID_Gains[3,:],[phi_des-phi, self.phi_integral, ang_vel_x_des-ang_vel_x])
        M_y = np.dot(self.PID_Gains[4,:],[theta_des-theta, self.theta_integral, ang_vel_y_des-ang_vel_y])
        M_z = np.dot(self.PID_Gains[5,:],[psi_des-psi, self.psi_integral, ang_vel_z_des-ang_vel_z])

        F_vec = np.array([T,M_x,M_y,M_z]).reshape(4,1)
        x = np.linalg.solve(self.motor_matrix, F_vec)

        m = self.sim_params["quadcopter"]["mass"]
        g = self.sim_params["constants"]["acc_gravity"]
        kf = self.params["quadcopter"]["motor"]["kf"]
        u_hover = ((1/kf)*(m*g)/4)*np.ones([4,1])       

        u = u_hover + x
        return u.squeeze()