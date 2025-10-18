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

        self.phi_des = 0
        self.theta_des = 0

        self.sim_dt = self.sim_params["time"]["dt"]
        if (self.sim_dt > self.outer_loop_dt) or (self.sim_dt > self.inner_loop_dt):
              print("Warning: Simulation time step greater than PID inner/outer loop time step")
        self.controller_dt = self.inner_loop_dt


        kf = self.sim_params["quadcopter"]["motor"]["kf"]
        km = self.sim_params["quadcopter"]["motor"]["km"]
        arm_length = self.sim_params["quadcopter"]["arm_length"]
        self.motor_matrix = np.array([
                [kf,                kf,            kf,                  kf],
                [0,                 arm_length*kf, 0,                   -arm_length*kf],
                [-arm_length*kf,    0,             arm_length*kf,       0],
                [km,                -km,           km,                  -km]
        ]) 

        m =  self.sim_params["quadcopter"]["mass"]
        g =  self.sim_params["constants"]["acc_gravity"]
        kf = self.sim_params["quadcopter"]["motor"]["kf"]
        self.u_hover = ((1/kf)*(m*g)/4)*np.ones([4,1])            


    def set_trajectory(self, trajectory):
        self.trajectory_object = trajectory

    def calculate_control(self,state,t): 

        pos_des, vel_des, acc_des = self.trajectory_object.evaluate_trajectory(t)
        if not isinstance(pos_des, np.ndarray):
             return self.u_hover.squeeze()

        u = self.position_controller(state, pos_des, vel_des, t)

        return u
    
    def position_controller(self, state, pos_des, vel_des, t):

        x, y, z    = state[0:3]
        vx, vy, vz = state[7:10]
        x_des, y_des, z_des    = pos_des
        vx_des, vy_des, vz_des = vel_des
        
        x_gains = [self.PID_Gains["outer_loop"]["proportional"]["x"], self.PID_Gains["outer_loop"]["integral"]["x"], self.PID_Gains["outer_loop"]["derivative"]["x"]]
        y_gains = [self.PID_Gains["outer_loop"]["proportional"]["y"], self.PID_Gains["outer_loop"]["integral"]["y"], self.PID_Gains["outer_loop"]["derivative"]["y"]]

        if t >= self.outer_loop_time:
            self.outer_loop_time += self.outer_loop_dt

            #TODO: Should this be inside or outside the loop?
            self.x_integral += self.sim_dt*(x_des-x)
            self.y_integral += self.sim_dt*(y-y_des)
            self.phi_des= np.dot(y_gains,[y-y_des, self.y_integral, vy-vy_des])
            if self.phi_des > 20:
                    self.phi_des = 20
            elif self.phi_des < -20:
                    self.phi_des = -20

            self.theta_des = np.dot(x_gains,[x_des-x, self.x_integral, vx_des-vx])
            if self.theta_des > 20:
                    self.theta_des = 20
            elif self.theta_des < -20:
                    self.theta_des = -20

            self.t_outer = 0

        u = self.attitude_controller(state, t, z, vz, z_des, vz_des)
        return u
  

    def attitude_controller(self, state, t, z, vz, z_des, vz_des):
  
        quaternion = R.from_quat(np.array(state[3:7]),scalar_first=True)
        eul = quaternion.as_euler('zyx', degrees=True)

        phi, theta, psi = [eul[2],eul[1],eul[0]]
        ang_vel_x, ang_vel_y, ang_vel_z = state[10:]

        psi_des = 0 #TODO: reexamine this later
        ang_vel_x_des = ang_vel_y_des = ang_vel_z_des = 0   #TODO: reexamine this later
        # ang_vel_x, ang_vel_x_des = state[10], self.des_state[9,i]    #using omega from state is an approximation

        #TODO: Should this be inside or outside the loop?
        self.z_integral     += self.sim_dt*(z_des-z)
        self.phi_integral   += self.sim_dt*(self.phi_des-phi)
        self.theta_integral += self.sim_dt*(self.theta_des-theta)
        self.psi_integral   += self.sim_dt*(psi_des-psi)

        z_gains     = [self.PID_Gains["outer_loop"]["proportional"]["z"], self.PID_Gains["outer_loop"]["integral"]["z"], self.PID_Gains["outer_loop"]["derivative"]["z"]]

        phi_gains   = [self.PID_Gains["inner_loop"]["proportional"]["phi"], self.PID_Gains["inner_loop"]["integral"]["phi"], self.PID_Gains["inner_loop"]["derivative"]["phi"]]
        theta_gains = [self.PID_Gains["inner_loop"]["proportional"]["theta"], self.PID_Gains["inner_loop"]["integral"]["theta"], self.PID_Gains["inner_loop"]["derivative"]["theta"]]
        psi_gains   = [self.PID_Gains["inner_loop"]["proportional"]["psi"], self.PID_Gains["inner_loop"]["integral"]["psi"], self.PID_Gains["inner_loop"]["derivative"]["psi"]]

        T   = np.dot(z_gains,    [z_des-z, self.z_integral, vz_des-vz])
        M_x = np.dot(phi_gains,  [self.phi_des-phi, self.phi_integral, ang_vel_x_des-ang_vel_x])
        M_y = np.dot(theta_gains,[self.theta_des-theta, self.theta_integral, ang_vel_y_des-ang_vel_y])
        M_z = np.dot(psi_gains,  [psi_des-psi, self.psi_integral, ang_vel_z_des-ang_vel_z])

        F_vec = np.array([T,M_x,M_y,M_z]).reshape(4,1)
        u_diff = np.linalg.solve(self.motor_matrix, F_vec)

        u = self.u_hover + u_diff
        return u.squeeze()