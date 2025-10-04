import numpy as np
from scipy.spatial.transform import Rotation as R

class Simulator():

    def __init__(self, cfg):
        self.params = cfg
        self.pad_matrix = np.block([
                        [np.zeros([1,3])],
                        [np.eye(3)]
                        ])    

    def simulate(self,controller): #-> str:
        """Simulate Quadcopter Flight"""

        dt = self.params["time"]["dt"]
        tf = self.params["time"]["duration"]
        waypoints =  self.params["world"]["waypoints"]

        n_steps = int(tf/dt)
        x0 = np.array(waypoints[0]["pose"], dtype=float)

        states   = np.zeros([13, n_steps+1])
        times    = np.zeros([n_steps+1])
        controls = np.zeros([4, n_steps])

        states[:, 0] = x0

        for idx in range(1,n_steps+1):
            x_current = states[:, idx-1]
            t  = idx * dt
            u  = controller.calculate_control(x_current, t) 
            
            if (t%1==0):
                print(f"Time: {t}")

            #TODO: CLIP CONTROL TO RESPECT LIMITS
                #####

            states[:, idx]  = self.take_rk4_step(x_current,u)
            times[idx]      = t
            controls[:,idx-1] = u


        return times, states, controls
        
    # def take_step(self):
    #     """ Simulate for a single time step"""

    def take_rk4_step(self, x_current, u):
        """ Numerical Integration with RK4 for a time step"""

        dt = self.params["time"]["dt"]
        
        #RK4 integration with zero-order hold on u
        k1 = self.quad_dynamics(x_current,u)
        k2 = self.quad_dynamics(x_current + (0.5*dt*k1), u)
        k3 = self.quad_dynamics(x_current + (0.5*dt*k2), u)
        k4 = self.quad_dynamics(x_current + (dt*k3)    , u)

        x = x_current + (
            (dt/6)*( k1 + (2*k2) + (2*k3) + k4  )
        )

        #re-normalize quaternion 
        x[3:7] = x[3:7]/np.linalg.norm(x[3:7])

        return x

    def quad_dynamics(self, x_current, u):
        """Calculate x_dot based on equations of motion"""
        
        position     = x_current[0:3]
        orientation  = x_current[3:7]
        velocity     = x_current[7:10]
        ang_velocity = x_current[10:]

        x_dot = np.zeros([13])
        rotation_matrix = R.from_quat(orientation,scalar_first=True).as_matrix()

        x_dot[0:3] = rotation_matrix@velocity
        x_dot[3:7] = 0.5*self.quaternion_multiply_left(orientation)@self.pad_matrix@ang_velocity

        mass = self.params["quadcopter"]["mass"]
        I_xx = self.params["quadcopter"]["I_xx"]
        I_yy = self.params["quadcopter"]["I_yy"]
        I_zz = self.params["quadcopter"]["I_zz"]
        arm_length = self.params["quadcopter"]["arm_length"]
        kf = self.params["quadcopter"]["motor"]["kf"]
        km = self.params["quadcopter"]["motor"]["km"]
        g = self.params["constants"]["acc_gravity"]

        I = np.array([
                [I_xx, 0,    0],
                [0,    I_yy, 0],
                [0,    0,    I_zz]
        ])

        u_matrix = np.block([
                [np.zeros([2,4])],
                [kf*np.ones([1,4])]
        ])  

        x_dot[7:10] = ( (rotation_matrix.T) @ np.array([0,0,-g]) ) + ( (1/mass)*(u_matrix@u) )  
        - ( self.hat_operator(ang_velocity) @ velocity )

        torques_body_frame = np.array([
            arm_length*kf*(u[1]-u[3]),
            arm_length*kf*(u[2]-u[0]),
            km*(u[0]-u[1]+u[2]-u[3])
            ])
        
        x_dot[10:] = np.linalg.solve(I,
        torques_body_frame - (self.hat_operator(ang_velocity)@I@ang_velocity)
        )  

        return x_dot

 
    def quaternion_multiply_left(self, q):
        """ Quaternion multiplication via left sided matrix multiplication """

        s = q[0]
        v = np.array(q[1:]).reshape((3,1))
        v_t = v.T

        Lq = np.block([
                [s,-v_t],
                [v, (s*np.eye(3))+self.hat_operator(v)]
        ])
        return Lq
    

    @staticmethod
    def hat_operator(x):
            # Takes a vector and returns 3x3 skew symmetric matrix
            x = np.array(x).reshape(3,)
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]

            return np.array([
                    [0,-x3,x2],
                    [x3,0,-x1],
                    [-x2,x1,0]
            ])