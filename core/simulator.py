import numpy as np
from scipy.spatial.transform import Rotation as R

class Simulator():

    def __init__(self, cfg):
        self.params = cfg
        self.pad_matrix = np.block([
                        [np.zeros([1,3])],
                        [np.eye(3)]
                        ])    
        self.actuator_limit = (self.params["quadcopter"]["limits"]["clip_factor"])*self.params["quadcopter"]["mass"]*self.params["constants"]["acc_gravity"]

        if (self.params["world"]["wind"]["active"] == "True"):
            self.wind_sim     = True  
            self.wind_vector  = np.array(self.params["world"]["wind"]["speed_vector"])
            self.wind_std_dev = np.array(self.params["world"]["wind"]["std_deviation"])
            #TODO: Clean this up later
            self.Cd = self.params["world"]["wind"]["Cd"]
            self.mass = self.params["quadcopter"]["mass"]
            self.arm_length = self.params["quadcopter"]["arm_length"]
        else:
            self.wind_sim     = False

    def simulate(self, controller, controller_dt): #-> str:
        """Simulate Quadcopter Flight"""

        dt = self.params["time"]["dt"]
        tf = self.params["time"]["duration"]
        input_delay = dt * self.params["time"]["delay_time_step"]
        waypoints =  self.params["world"]["waypoints"]

        n_steps = int(tf/dt)
        x0 = np.array(waypoints[0]["pose"], dtype=float)

        states   = np.zeros([13, n_steps+1])
        times    = np.zeros([n_steps+1])
        controls = np.zeros([4, n_steps])

        states[:, 0]    = x0
        controller_time = 0
        u_calculated = False  # To prevent u_calc being recalculated for the entire window [controller_time, controller_time+input_delay)

        print("Started Sim Loop")
        for idx in range(1,n_steps+1):
            x_current = states[:, idx-1]
            t  = idx * dt        

            if (t>=controller_time) and (t <= (controller_time+input_delay) ):
                if u_calculated == False:
                    u_calc  = controller.calculate_control(x_current, t) 
                    u_calculated = True
            if (t >= (controller_time+input_delay) ) or (idx==1):
                if input_delay == 0:
                    u_calc  = controller.calculate_control(x_current, t) 
                u = u_calc
                controller_time += controller_dt
                u_calculated = False
                print(f"Time: {t}, Control Input given")
            
            if (t%1==0):
                print(f"Time: {t}")

            # Force actuator limits to be respected
            u = np.clip(u, 0, self.actuator_limit)

            # print(f"x_current: {x_current}")
            # print(f"Time: {t}")

            states[:, idx]  = self.take_rk4_step(x_current,u)
            times[idx]      = t
            controls[:,idx-1] = u

            #TODO: Check if crashed, integrate sensor noise

        return times, states, controls
        
    def take_rk4_step(self, x_current, u):
        """ Numerical Integration with RK4 for a time step"""

        dt = self.params["time"]["dt"]

        if self.wind_sim == True:
            wind_acc = self.wind_dynamics(x_current)   
        else:
            wind_acc = 0
        
        #RK4 integration with zero-order hold on u
        k1 = self.quad_dynamics(x_current,u, wind_acc)
        k2 = self.quad_dynamics(x_current + (0.5*dt*k1), u, wind_acc)
        k3 = self.quad_dynamics(x_current + (0.5*dt*k2), u, wind_acc)
        k4 = self.quad_dynamics(x_current + (dt*k3)    , u, wind_acc)

        x = x_current + (
            (dt/6)*( k1 + (2*k2) + (2*k3) + k4  )
        )

        #re-normalize quaternion 

        x[3:7] = x[3:7]/np.linalg.norm(x[3:7])

        return x

    def quad_dynamics(self, x_current, u, wind_acc):
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

        x_dot[7:10] = ( (rotation_matrix.T) @ np.array([0,0,-g]) ) + ( (1/mass)*(u_matrix@u) )  - ( self.hat_operator(ang_velocity) @ velocity )

        if self.wind_sim == True:
            x_dot[7:10] += wind_acc

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


    def wind_dynamics(self, x_current):
        orientation  = x_current[3:7]
        velocity     = x_current[7:10]
        rotation_matrix = R.from_quat(orientation,scalar_first=True).as_matrix()

        if (self.params["world"]["wind"]["type"] == "constant"):
            rel_velocity = velocity - (rotation_matrix.T@self.wind_vector)
        else:
            rng = np.random.default_rng()
            gaussian_wind_vec = rng.normal(loc=self.wind_vector, scale=self.wind_std_dev)
            rel_velocity = velocity - (rotation_matrix.T@gaussian_wind_vec)

        # Modelling wind force as -0.5*pho*Cd*A*v_rel^2 with area approximated as arm_length^2
        return (-1/self.mass)*(0.5*1.225*self.Cd)*(self.arm_length*self.arm_length)*rel_velocity*np.abs(rel_velocity)
    
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