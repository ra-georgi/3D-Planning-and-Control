from controllers.controller import Controller
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import block_diag
import jax
import jax.numpy as jnp

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

        self.mass =       self.sim_params["quadcopter"]["mass"]
        self.I_xx =       self.sim_params["quadcopter"]["I_xx"]
        self.I_yy =       self.sim_params["quadcopter"]["I_yy"]
        self.I_zz =       self.sim_params["quadcopter"]["I_zz"]
        self.arm_length = self.sim_params["quadcopter"]["arm_length"]
        self.kf =         self.sim_params["quadcopter"]["motor"]["kf"]
        self.km =         self.sim_params["quadcopter"]["motor"]["km"]
        self.g =          self.sim_params["constants"]["acc_gravity"]
        self.dt =         self.sim_params["time"]["dt"]

        self.pad_matrix = jnp.block([
                        [jnp.zeros([1,3])],
                        [jnp.eye(3)]
                        ])  


    def calculate_control(self,state,t): 

        return u
    


    def set_trajectory(self, trajectory):
        # self.trajectory_object = trajectory
        self.lqr_K = self.calculate_gains(trajectory)
        pass

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

        for k in range(num_steps-2,-1,-1):
            gains_dict = {}
            gains_dict["t"]  = times[k]
            
            pos_des, vel_des, acc_des = trajectory_object.evaluate_trajectory(times[k])
            jerk_des, snap_des        = trajectory_object.evaluate_jerk_snap(times[k])
            full_ref_state, ref_control = self.get_full_references(pos_des, vel_des, acc_des, jerk_des, snap_des)
            #TODO: Convert above output to appropriate numpy arrays

            # linearize about state to get A and B
            A = np.asarray(jax.jacfwd(lambda y: self.take_rk4_step(y, ref_control))(full_ref_state))
            B = np.asarray(jax.jacfwd(lambda y: self.take_rk4_step(full_ref_state,y))(ref_control))

             # get next reference quaternion for jacobian modification
            pos_d, vel_d, acc_d = trajectory_object.evaluate_trajectory(times[k+1])
            jerk_d, snap_d      = trajectory_object.evaluate_jerk_snap(times[k+1])
            next_ref_state, _   = self.get_full_references(pos_d, vel_d, acc_d, jerk_d, snap_d)

            #apply modification for quaternions
            A_mod = (self.E(next_ref_state[3:7]).T)@A@self.E(full_ref_state[3:7])
            B_mod = (self.E(next_ref_state[3:7]).T)@B

            K = np.linalg.solve( (self.R + (B_mod.T@P[k+1,:,:]@B_mod) ), (B_mod.T @ P[k+1,:,:] @ A_mod) )
            P[k,:,:] = self.Q + ( A_mod.T @ P[k+1,:,:] @ (A_mod - B_mod @ K)    )
            gains_dict["K"]  = K
            lqr_gains.append(gains_dict)

        return lqr_gains


    def get_full_references(self, pos_des, vel_des, acc_des, jerk_des, snap_des):
        # Creates a full state reference and a control reference based on approximations to dynamics

        phi_des   = -acc_des[1]/self.g
        theta_des =  acc_des[0]/self.g
        psi_des   =  0

        r    =  R.from_euler('ZYX', [[psi_des, theta_des, phi_des]])
        quat =  r.as_quat(scalar_first=True)

        wx_des = -jerk_des[1]/self.g
        wy_des =  jerk_des[0]/self.g
        wz_des =  0

        phi_dot_dot   = -snap_des[1]/self.g
        theta_dot_dot =  snap_des[0]/self.g
        
        tau_phi   = (self.I_xx*phi_dot_dot) + (wy_des*wz_des)*(self.I_zz-self.I_yy)
        tau_theta = (self.I_yy*theta_dot_dot) + (wx_des*wz_des)*(self.I_xx-self.I_zz)
        tau_psi =   (wx_des*wy_des)*(self.I_yy-self.I_xx)

        T_des = self.mass*(acc_des[2]+self.g)

        state = np.array([pos_des[0], pos_des[1], pos_des[2],
                          quat[0][0], quat[0][1], quat[0][2], quat[0][3],
                          vel_des[0], vel_des[1], vel_des[2], 
                          wx_des,wy_des,wz_des])

        # state =   [pos_des,quat,vel_des,wx_des,wy_des,wz_des]
        control = np.array([T_des,tau_phi,tau_theta,tau_psi ])

        return state, control
    

    def take_rk4_step(self, x_current, u):
        """ Numerical Integration with RK4 for a time step"""
        
        # Ensure inputs are JAX arrays
        x_current = jnp.asarray(x_current)
        u = jnp.asarray(u)

        #RK4 integration with zero-order hold on u  
        k1 = self.quad_dynamics(x_current,u)
        k2 = self.quad_dynamics(x_current + (0.5*self.dt*k1), u)
        k3 = self.quad_dynamics(x_current + (0.5*self.dt*k2), u)
        k4 = self.quad_dynamics(x_current + (self.dt*k3)    , u)

        x = x_current + (
            (self.dt/6)*( k1 + (2*k2) + (2*k3) + k4  )
        )

        #re-normalize quaternion 
        # x[3:7] = x[3:7]/np.linalg.norm(x[3:7])
        norm_var = jnp.linalg.norm(x[3:7])
        x = x.at[3:7].set(x[3:7]/norm_var)

        return x
    

    def quad_dynamics(self, x_current, u):
        """Calculate x_dot based on equations of motion"""
        
        position     = x_current[0:3]
        orientation  = x_current[3:7]
        velocity     = x_current[7:10]
        ang_velocity = x_current[10:]
        
        # x_dot = jnp.zeros([13])
        rotation_matrix = self.quat_to_rotmat(orientation)

        pos_dot = rotation_matrix@velocity
        quat_dot = 0.5*self.quaternion_multiply_left(orientation)@self.pad_matrix@ang_velocity

        I = jnp.array([
                [self.I_xx, 0,         0],
                [0,         self.I_yy, 0],
                [0,         0,         self.I_zz]
        ])

        u_matrix = jnp.block([
                [jnp.zeros([2,4])],
                [self.kf*jnp.ones([1,4])]
        ])  

        vel_dot = ( (rotation_matrix.T) @ jnp.array([0,0,-self.g]) ) + ( (1/self.mass)*(u_matrix@u) )  
        - ( self.hat_operator(ang_velocity) @ velocity )

        torques_body_frame = jnp.array([
            self.arm_length*self.kf*(u[1]-u[3]),
            self.arm_length*self.kf*(u[2]-u[0]),
            self.km*(u[0]-u[1]+u[2]-u[3])
            ])
        
        omega_dot = jnp.linalg.solve(I,
        torques_body_frame - (self.hat_operator(ang_velocity)@I@ang_velocity)
        )  

        x_dot = jnp.block([
            pos_dot,quat_dot,vel_dot,omega_dot
        ])

        return x_dot
    
    def quat_to_rotmat(self,q):    
        # Converts quaternion to rotation matrix
            
        H = jnp.block([
                [jnp.zeros([1,3])],
                [jnp.eye(3)]
        ]
        )

        Lq = self.quaternion_multiply_left(q)
        Rq = self.quaternion_multiply_right(q)

        Q = (H.T)@Lq@(Rq.T)@H
        # Q = (H.T)@T@(self.L(q))@T@(self.L(q))@H
        return Q     

    def quaternion_multiply_left(self, q):
        """ Quaternion multiplication via left sided matrix multiplication """

        s = q[0]
        v = jnp.array(q[1:]).reshape((3,1))
        v_t = v.T

        Lq = jnp.block([
                [s,-v_t],
                [v, (s*jnp.eye(3))+self.hat_operator(v)]
        ])
        return Lq
    
    def quaternion_multiply_right(self,q):
        #Takes a quaternion and returns a matrix for right multiplication
        s = q[0]
        v = jnp.array(q[1:]).reshape((3,1))
        v_t = v.T

        Rq = jnp.block([
                [s,-v_t],
                [v, (s*jnp.eye(3))-self.hat_operator(v)]
        ])
        return Rq

    @staticmethod
    def hat_operator(x):
        # Takes a vector and returns 3x3 skew symmetric matrix
        x = jnp.array(x).reshape(3,)
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]   

        return jnp.array([
                [0,-x3,x2],
                [x3,0,-x1],
                [-x2,x1,0]
        ])
    
    def E(self,q):
            return block_diag(np.eye(3), self.calc_attitude_jacobian(q), np.eye(6))
    

    def calc_attitude_jacobian(self,q):
            # Calculate attitude jacobian at q
            H = jnp.block([
                    [jnp.zeros([1,3])],
                    [jnp.eye(3)]
            ])
            G = self.quaternion_multiply_left(q)@H
            return G