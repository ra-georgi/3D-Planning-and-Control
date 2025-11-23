from controllers.controller import Controller
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import block_diag
import casadi as ca
import jax
import jax.numpy as jnp


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

        self.pad_matrix = jnp.block([
                        [jnp.zeros([1,3])],
                        [jnp.eye(3)]
                        ])  

        self.motor_matrix = np.array([
                [self.kf,                     self.kf,                 self.kf,                  self.kf],
                [0,                           self.arm_length*self.kf, 0,                        -self.arm_length*self.kf],
                [-self.arm_length*self.kf,    0,                       self.arm_length*self.kf,       0],
                [self.km,                     -self.km,                self.km,                  -self.km]
        ]) 

        self.mpc_horizon = self.controller_params["mpc_horizon"]
        self.actuator_limit = (self.sim_params["quadcopter"]["limits"]["clip_factor"])*self.sim_params["quadcopter"]["mass"]*self.sim_params["constants"]["acc_gravity"]
        self.u_min = np.array([0,0,0,0])
        self.u_max = self.actuator_limit*np.ones(4)

 

    def set_trajectory(self, trajectory):
        self.trajectory_object = trajectory

    def calculate_control(self,state,t): 
        
        times = []
        ref_states_quat = np.zeros((13, self.mpc_horizon))
        ref_control     = np.zeros(( 4, self.mpc_horizon))           # Is this a must, can we do u.T@R@u
        
        A_reduced       = np.zeros((self.mpc_horizon, 12, 12))
        B_reduced       = np.zeros((self.mpc_horizon, 12,  4))

        for i in range(self.mpc_horizon):
            times.append(t + (i*self.controller_dt))

        for index, time in enumerate(times):
            pos_des, vel_des, acc_des = self.trajectory_object.evaluate_trajectory(time)

            # If time is outside of planned trajectory
            if not isinstance(pos_des, np.ndarray):
                pos_des = self.waypoints[-1]["pose"][0:3]
                ref_states_quat[:,index] = np.array([pos_des[0], pos_des[1], pos_des[2],
                            1, 0, 0, 0,
                            0, 0, 0, 
                            0, 0, 0])
                u = (1/self.kf)*(self.mass*self.g)/4
                ref_control[:,index] = u*np.ones((4))
            else:
                ref_states_quat[:,index], ref_control[:,index] = self.get_full_reference_estimate(pos_des, vel_des, acc_des)

            # linearize about state to get A and B
            A = np.asarray(jax.jacfwd(lambda y: self.take_rk4_step(y, ref_control[:,index]))(ref_states_quat[:,index]))
            B = np.asarray(jax.jacfwd(lambda y: self.take_rk4_step(ref_states_quat[:,index], y))(ref_control[:,index]))

             # get next reference quaternion for jacobian modification
            pos_d, vel_d, acc_d = self.trajectory_object.evaluate_trajectory(time + self.controller_dt)
            if not isinstance(pos_d, np.ndarray):
                pos_des = self.waypoints[-1]["pose"][0:3]
                next_ref_state = np.array([pos_des[0], pos_des[1], pos_des[2],
                            1, 0, 0, 0,
                            0, 0, 0, 
                            0, 0, 0])
            else:
                next_ref_state, _   = self.get_full_reference_estimate(pos_d, vel_d, acc_d)

            #apply modification for quaternions
            A_reduced[index,:,:] = (self.E(next_ref_state[3:7]).T)@A@self.E(ref_states_quat[:,index][3:7])
            B_reduced[index,:,:] = (self.E(next_ref_state[3:7]).T)@B

        ref_states_mrp = np.zeros((12, self.mpc_horizon))
        for i in range(self.mpc_horizon):
            mrp = self.quat_to_rodrig(ref_states_quat[3:7,i])
            ref_states_mrp[:,i] = np.concatenate((ref_states_quat[0:3, i], mrp, ref_states_quat[7:10, i],ref_states_quat[10:13, i]), axis=None)      

        # Create Casadi optimization problem
        opti = ca.Opti('conic')
        o_del_x = opti.variable(12, self.mpc_horizon)        
        o_del_u = opti.variable( 4, self.mpc_horizon-1)

        # Parameters
        o_x0   = opti.parameter(12)                      # initial state
        o_xref = opti.parameter(12, self.mpc_horizon)    # reference trajectory        

        # Initial condition constraint
        opti.subject_to(o_del_x[:, 0] == o_x0)        

        J = 0

        for k in range(self.mpc_horizon-1):
            e = o_del_x[:, k] - o_xref[:, k]
            J += 0.5 * ca.dot(e, self.Q @ e) + ( 0.5 * ca.dot(o_del_u[:, k], self.R@o_del_u[:, k])  )

            # linear dynamics
            opti.subject_to(o_del_x[:, k+1] == (A_reduced[k,:,:] @ o_del_x[:, k]) + (B_reduced[k,:,:] @ o_del_u[:, k])  )

            # input box constraints
            opti.subject_to(self.u_min    <= (o_del_u[:, k]+ref_control[:,k]) )
            opti.subject_to( (o_del_u[:, k]+ ref_control[:,k]) <= self.u_max)

        # terminal cost
        eN = o_del_x[:, self.mpc_horizon-1] - o_xref[:, self.mpc_horizon-1]
        J += 0.5 * ca.dot(eN, self.Qf @ eN)

        opti.minimize(J)

        # choose a conic/QP solver (e.g. OSQP)
        p_opts = {"expand": True}  # optional but often good for QPs
        s_opts = {"verbose": False}
        opti.solver('osqp', p_opts, s_opts)

        mrp = self.quat_to_rodrig(state[3:7])
        init_state_reduced = np.concatenate((state[0:3], mrp, state[7:10],state[10:13]), axis=None)

        opti.set_value(o_x0,   init_state_reduced)
        opti.set_value(o_xref, ref_states_mrp)

        sol = opti.solve()
        U_opt = sol.value(o_del_u)
        # X_opt = sol.value(X)
        del_u = U_opt[:, 0]

        return del_u + ref_control[:,0]
    

    def get_full_reference_estimate(self, pos_des, vel_des, acc_des):
        # Creates a full state reference and a control reference estimate

        phi_des   = np.degrees(-acc_des[1]/self.g)
        if phi_des > 15:           # To stay close to small angle approximation
            phi_des = 15
        elif phi_des < -15:
            phi_des = -15
        theta_des =  np.degrees(acc_des[0]/self.g)
        if theta_des > 15:           # To stay close to small angle approximation
            theta_des = 15
        elif theta_des < -15:
            theta_des = -15
        psi_des   =  0

        r    =  R.from_euler('ZYX', [psi_des, theta_des, phi_des], degrees=True)
        quat =  r.as_quat(scalar_first=True)
        R_wb = self.quat_to_rotmat(quat)     

        vel_des_body = R_wb.T @ vel_des  

        wx_des = 1*(phi_des/15)
        wy_des = 1*(theta_des/15)
        wz_des =  0

        tau_phi   = 0 #0.1*wx_des
        tau_theta = 0 #0.1*wy_des
        tau_psi   = 0
        T_des = self.mass*(acc_des[2]+self.g)

        F_vec = np.array([T_des,tau_phi,tau_theta,tau_psi ]).reshape(4,1)
        u_diff = np.linalg.solve(self.motor_matrix, F_vec)

        # u = self.u_hover + u_diff
        control = u_diff

        state = np.array([pos_des[0], pos_des[1], pos_des[2],
                          quat[0], quat[1], quat[2], quat[3],
                          vel_des_body[0], vel_des_body[1], vel_des_body[2],
                          wx_des,wy_des,wz_des])        
        
        return state, control.squeeze()
    
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
    
    @staticmethod
    def rodrig_to_quat(phi):
            phi = np.array(phi).reshape((3,1))
            a = 1/np.sqrt(phi.T@phi)
            return a*np.block([ [1],[phi] ])

    @staticmethod
    def quat_to_rodrig(quat):
            quat = np.array(quat).reshape((4,))
            return quat[1:4]/quat[0]
    

    def take_rk4_step(self, x_current, u):
        """ Numerical Integration with RK4 for a time step"""
        
        # Ensure inputs are JAX arrays
        x_current = jnp.asarray(x_current)
        u = jnp.asarray(u)

        #RK4 integration with zero-order hold on u  
        k1 = self.quad_dynamics(x_current,u)
        k2 = self.quad_dynamics(x_current + (0.5*self.controller_dt*k1), u)
        k3 = self.quad_dynamics(x_current + (0.5*self.controller_dt*k2), u)
        k4 = self.quad_dynamics(x_current + (self.controller_dt*k3)    , u)

        x = x_current + (
            (self.controller_dt/6)*( k1 + (2*k2) + (2*k3) + k4  )
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

        vel_dot = ( (rotation_matrix.T) @ jnp.array([0,0,-self.g]) ) + ( (1/self.mass)*(u_matrix@u) )  - ( self.hat_operator(ang_velocity) @ velocity )

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