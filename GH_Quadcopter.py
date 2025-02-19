import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
from scipy.linalg import block_diag
import math
import jax
import jax.numpy as jnp
import control as ct

# from Indirect_TrajOpt import iLQR
from Indirect_TrajOpt_Quat import iLQR

import cvxpy as cp



class Quadcopter:
        
        # Constructor
        def __init__(self, param_dict) -> None:
                self.m = param_dict["Mass"]
                self.Ixx = param_dict["Ixx"]
                self.Iyy = param_dict["Iyy"]
                self.Izz = param_dict["Izz"]
                self.l = param_dict["ArmLength"]
                self.Kf = param_dict["Kf"]
                self.Km = param_dict["Km"]
                self.g = param_dict["Accel_g"]
                self.h = param_dict["TimeStep"]
                self.tf = param_dict["FinalTime"]

                self.control_lim = False
                self.CA = False
                self.CBF_active = False

                self.param_dict = param_dict
                self.N = int(self.tf/self.h)+1  
                # self.PID_Gains =  np.zeros((6,3)) 
                
        def setup_sim(self,sim_params):
                if sim_params[0]:
                        self.CA = True
                        self.CBF_active = sim_params[1][0]
                        self.obs_coords = sim_params[1][1]

                if sim_params[2]:
                        self.control_lim = True
                        self.clip_factor = sim_params[3]

                self.motor_matrix = np.array([
                        [self.Kf,         self.Kf,        self.Kf,         self.Kf],
                        [0,               self.l*self.Kf, 0,               -self.l*self.Kf],
                        [-self.l*self.Kf, 0,              self.l*self.Kf,  0],
                        [self.Km,         -self.Km,       self.Km,         -self.Km]
                ]) 

        # Simulate
        def simulate(self,initial_cond,control_strat):
                # METHOD1 Summary of this method goes here 
                #   Detailed explanation goes here
                t = np.linspace(0, self.tf, num=self.N)
                state = np.zeros([13, self.N])
                
                r = R.from_euler('xyz', [initial_cond[3], initial_cond[4], initial_cond[5]], degrees=True)
                quat = r.as_quat()

                #For Plotting
                self.xic_pos = initial_cond[0:3]
                
                state[0:3,0] = initial_cond[0:3]
                state[3:7,0] = [quat[3],quat[0],quat[1],quat[2]]
                state[7:10,0] = initial_cond[6:9]
                state[10:13,0] = initial_cond[9:]           
                
                for i in range(self.N-1):
                        # control_list = self.DIRCOL_control[i]
                        # u =  jnp.array([control_list[0],control_list[1],control_list[2],control_list[3]]).squeeze()

                        control_list = control_strat(state[:,i],t[i],i)
                        u =  jnp.array([control_list[0],control_list[1],control_list[2],control_list[3]]).squeeze()
                        # print(f"u trajectory {u}")

                        # Control Barrier Function for obstacles
                        if self.CBF_active:
                                u = self.CBF(state[:,i],u)
                                # print(f"u cbf {u}")

                        if self.control_lim:
                                u = np.clip(u, 0, self.m*self.g*self.clip_factor)
                                # print(f"u clipped {u}")

                        state[:,i+1] = self.quad_rk4_step(state[:,i],u).reshape([13])
                        if jnp.linalg.norm(state[3:7,i+1]) < 1.0:
                                state[3:7,i+1] = state[3:7,i+1]/jnp.linalg.norm(state[3:7,i+1])
                        # print(state[:,i+1])
                        if ( max(state[:,i+1]) > 1000000 ):
                                t = t[i]
                                print("Numerical issues")
                                break

                return state, t
        
        # Generate Plots
        def quad_plots(self, state, t):

                new_state = np.zeros([12, len(t)])
                new_state[0:3,:] = state[0:3,:]
                new_state[6:9,:] = state[7:10,:]
                new_state[9:12,:] = state[10:13,:]

                for i in range(len(t)):
                        quaternion = np.array(state[3:7,i]).reshape(4,1)
                        Q = self.quat_to_rotmat(quaternion)
                        new_state[6:9,i] = Q@new_state[6:9,i]

                        quat = R.from_quat([state[4,i], state[5,i], state[6,i], state[3,i]])
                        eul = quat.as_euler('xyz', degrees=True)
                        new_state[3:6,i] = [eul[0],eul[1],eul[2]]
                        # roll_x, pitch_y, yaw_z = self.euler_from_quaternion(state[4,i], state[5,i], state[6,i], state[3,i])
                        # new_state[3:6,i] = [math.degrees(roll_x),math.degrees(pitch_y),math.degrees(yaw_z)]

                num_rows = 4
                num_cols = 3
                # Create a figure with subplots
                fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))

                # Flatten the 2D array of subplots for easier indexing
                axs = axs.flatten()

                # Labels for the y-axis
                y_labels = [
                'x_pos', 'y_pos', 'z_pos',
                'phi_euler', 'theta_euler', 'psi_euler',
                'Vx', 'Vy', 'Vz',
                'AngVx_b', 'AngVy_b', 'AngVz_b'
                ]                

                for i, ax in enumerate(axs):
                        color = 'blue'
                        if i % num_cols == 1:
                                color = 'red'
                        elif i % num_cols == 2:
                                color = 'green'                        
                        ax.plot(t, new_state[i, :],color=color)
                        ax.set_xlabel('Time')
                        ax.set_ylabel(y_labels[i])

                # Add a single title for the entire window
                fig.suptitle('Quadcopter State Variables vs Time', fontsize=16)    
                plt.tight_layout()
                plt.show()                                   

        # Create animation
        def quad_animation(self,state,t):
                fig = plt.figure(figsize=(8,6))
                self.ax_anim = plt.axes(projection='3d')
                # Adjust the size of the plot within the figure
                plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

                self.ax2 = plt.axes([0.8, 0.7, 0.2, 0.2])
                self.ax2.axis('off')  # Hide the axes for the inset      

                # Initialize the text in the inset axes with a box around it
                bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray")
                self.text_t = self.ax2.text(0.1, 0.8, '', transform=self.ax2.transAxes, fontsize=12, bbox=bbox_props)
                self.text_x = self.ax2.text(0.1, 0.6, '', transform=self.ax2.transAxes, fontsize=12, bbox=bbox_props)   
                self.text_y = self.ax2.text(0.1, 0.4, '', transform=self.ax2.transAxes, fontsize=12, bbox=bbox_props)
                self.text_z = self.ax2.text(0.1, 0.2, '', transform=self.ax2.transAxes, fontsize=12, bbox=bbox_props)                                             
                
                x0, y0, z0 = state[0,0], state[1,0], state[2,0]
                x_min = np.min(state[0,:])
                y_min = np.min(state[1,:])
                z_min = np.min(state[2,:])
                x_max = np.max(state[0,:])
                y_max = np.max(state[1,:])
                z_max = np.max(state[2,:])

                # self.ax_anim.set_title(f"Time: {t[0]} s, Position {x0}, {y0}, {z0}")

                l = self.l
                self.quad_Arm1 = self.ax_anim.plot3D([x0+l, x0-l], [y0, y0], [z0, z0], lw=3 )[0]
                self.quad_Arm2 = self.ax_anim.plot3D([x0, x0], [y0+l, y0-l], [z0, z0], lw=3 )[0]
                self.quad_traj = self.ax_anim.plot3D(x0, y0, z0, 'gray')[0] 

                # ax_anim.set_xlim([x_min-1,x_max+1])
                # ax_anim.set_ylim([y_min-1,y_max+1])
                # ax_anim.set_zlim([z_min-1,z_max+1])

                #To make quadcopter's arms look equal in animation
                self.ax_anim.set_xlim([0,10])
                self.ax_anim.set_ylim([0,10])
                self.ax_anim.set_zlim([0,10])      

                # self.ax_anim.scatter(self.xic_pos[0], self.xic_pos[1], self.xic_pos[2])    
                # Plot start and goal points
                for wp in self.waypoints:
                        self.ax_anim.scatter(wp[0], wp[1], wp[2])     

                if self.CA:
                        r = 0.5
                        #Plot obstacles
                        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
                        x = r*np.cos(u)*np.sin(v)
                        y = r*np.sin(u)*np.sin(v) 
                        z = r*np.cos(v)
                        for obs in self.obs_coords:
                                self.ax_anim.plot_surface(x+obs[0], y+obs[1], z+obs[2], color='b',alpha=0.3)                  
     
                def update_anim_quad(frame, state, t, self):
                        # for each frame, update the data stored on each artist.

                        time = t[frame]
                        xt, yt, zt = state[0,frame], state[1,frame], state[2,frame]
                        quat = np.array(state[3:7,frame]).reshape(4,1)
                        Q = self.quat_to_rotmat(quat)

                        Arm1_Start = np.array([self.l,0,0])
                        Arm1_End = np.array([-self.l,0,0])

                        Arm2_Start = np.array([0,self.l,0])
                        Arm2_End = np.array([0,-self.l,0])

                        Arm1_Start = Q @ Arm1_Start
                        Arm1_End = Q @ Arm1_End
                        Arm2_Start = Q @ Arm2_Start
                        Arm2_End = Q @ Arm2_End              

                        self.quad_Arm1.set_data_3d([xt+Arm1_Start[0], xt+Arm1_End[0]], [yt+Arm1_Start[1], yt+Arm1_End[1]], [zt+Arm1_Start[2], zt+Arm1_End[2]])
                        self.quad_Arm2.set_data_3d([xt+Arm2_Start[0], xt+Arm2_End[0]], [yt+Arm2_Start[1], yt+Arm2_End[1]], [zt+Arm2_Start[2], zt+Arm2_End[2]])
                        self.quad_traj.set_data_3d(state[0,:frame],state[1,:frame],state[2,:frame])

                        self.text_t.set_text(f't = {time:.2f} s')
                        self.text_x.set_text(f'x = {xt:.2f} m')
                        self.text_y.set_text(f'y = {yt:.2f} m')
                        self.text_z.set_text(f'z = {zt:.2f} m')
                        # self.ax_anim.set_title(f"Time: {time} s, Position {xt}, {yt}, {zt}")
                        # print(f"Position at Frame no. {frame} is {xt},{yt},{zt}")
                        # self.fig.gca().relim()
                        # self.fig.gca().autoscale_view() 

                        return 

                self.ani = FuncAnimation(fig=fig, func=update_anim_quad,frames=state.shape[1], fargs=(state,t,self),interval=15)
                plt.show()

        def quad_rk4_step(self,x,u):
            
            h = self.h
            #RK4 integration with zero-order hold on u
            f1 = self.quad_dynamics(x, u)
            f2 = self.quad_dynamics(x + 0.5*h*f1, u)
            f3 = self.quad_dynamics(x + 0.5*h*f2, u)
            f4 = self.quad_dynamics(x + h*f3, u)
            xn = x + (h/6.0)*(f1 + 2*f2 + 2*f3 + f4)
            #re-normalize quaternion 
            norm_var = jnp.linalg.norm(xn[3:7])
            xn = xn.at[3:7].set(xn[3:7]/norm_var)      

            return xn        
        
        def quad_dynamics(self,x,u):
                # Takes in current state and returns xdot

                norm_var = jnp.linalg.norm(x[3:7])
                # x = x.at[3:7].set(x[3:7]/norm_var)
                # x.at[3].set(x[3]/norm_var)      
  
                quat = x[3:7]
                vel = x[7:10]
                omega = x[10:]

                Q = self.quat_to_rotmat(quat)
                pos_dot = Q@vel

                H = jnp.block([
                        [jnp.zeros([1,3])],
                        [jnp.eye(3)]
                ])        
                quat_dot = 0.5*self.L(quat)@H@omega

                m = self.m
                Kf = self.Kf
                Km = self.Km
                l = self.l
                g = self.g

                I = jnp.array([
                        [self.Ixx,0,0],
                        [0,self.Iyy,0],
                        [0,0,self.Izz]
                ])

                u_matrix = jnp.block([
                        [jnp.zeros([2,4])],
                        [Kf*jnp.ones([1,4])]
                ])                   

                vel_dot = ( (Q.T)@jnp.array([0,0,-g]) ) + ( (1/m)*(u_matrix@u) )  -( self.hat_operator(omega)@vel )

                tau_b = jnp.array([l*Kf*(u[1]-u[3]),l*Kf*(u[2]-u[0]),Km*(u[0]-u[1]+u[2]-u[3])])
                omega_dot=jnp.linalg.solve(I,tau_b - (self.hat_operator(omega)@I@omega))  
         
                return jnp.block([
                        pos_dot,quat_dot,vel_dot,omega_dot
                ])       

        def setup_PTP(self,way_points,inter_times,PTP_Params,ctrl_method):

                #for plotting
                self.waypoints = way_points
                self.inter_times = inter_times
                way_point_index = []

                if ctrl_method == "PID":
                        self.x_integral = 0      
                        self.y_integral = 0                          
                        self.z_integral = 0
                        self.phi_integral = 0
                        self.theta_integral = 0
                        self.psi_integral = 0

                        if not hasattr(self, 'PID_Gains'):   #Check if auto tuning was done
                                self.PID_Gains = PTP_Params

                        inner_loop_rate = 1/self.h
                        outer_loop_rate = inner_loop_rate/4
                        self.outer_loop_h = 1/outer_loop_rate
                        self.t_outer = self.outer_loop_h 

                        # self.des_state = np.zeros((12,self.N))

                        x_final =  np.array(way_points[-1], dtype =float)
                        self.des_state = np.kron(np.ones((1,self.N)), x_final.reshape((12,1)))
                        inter_points = np.array(way_points[1:-1])
                                
                        for i in inter_times:
                                index = int((i/self.h))
                                way_point_index.append(index)

                        index_array = way_point_index.copy()
                        index_array.insert(0,0)

                        for i in range(len(index_array)-1):
                                self.des_state[:,index_array[i]:index_array[i+1]+1] = inter_points[i,:].reshape(12,1)                       

                elif ctrl_method == "LQR":
                        quat_way_points = []
                        for i in way_points:
                                r = R.from_euler('xyz', [i[3], i[4], i[5]], degrees=True)
                                quat = r.as_quat()
                                quat_way_points.append(np.concatenate((i[0:3],[quat[3],quat[0],quat[1],quat[2]],i[6:9],i[9:]), axis=None))  

                        x_final =  np.array(quat_way_points[-1], dtype =float)
                        self.des_state = np.kron(np.ones((1,self.N)), x_final.reshape((13,1)))
                        inter_points = np.array(quat_way_points[1:-1])
                                
                        for i in inter_times:
                                index = int((i/self.h))
                                way_point_index.append(index)

                        self.index_array = way_point_index.copy()
                        self.index_array.insert(0,0)

                        for i in range(len(self.index_array)-1):
                                self.des_state[:,self.index_array[i]:self.index_array[i+1]+1] = inter_points[i,:].reshape(13,1)   

                        self.Q_lqr, self.R_lqr = PTP_Params    

                elif ctrl_method == "MPC":    
                        pass
                        self.Q_MPC = PTP_Params[0]
                        self.R_MPC = PTP_Params[1]
                        self.Qf_MPC = PTP_Params[2]
                        # self.N+MPC_Params[4]
                        self.MPC_Horizon = PTP_Params[3]
                        self.xmin = PTP_Params[4]
                        self.xmax = PTP_Params[5]

                        quat_way_points = []
                        for i in way_points:
                                r = R.from_euler('xyz', [i[3], i[4], i[5]], degrees=True)
                                quat = r.as_quat()
                                quat_way_points.append(np.concatenate((i[0:3],[quat[3],quat[0],quat[1],quat[2]],i[6:9],i[9:]), axis=None))  

                        reduced_way_points = []
                        for i in quat_way_points:
                                mrp = self.quat_to_rodrig(i[3:7])
                                reduced_way_points.append(np.concatenate((i[0:3],mrp,i[7:10],i[10:13]), axis=None))

                        x_final =  np.array(quat_way_points[-1], dtype =float)
                        self.des_state = np.kron(np.ones((1,self.N+self.MPC_Horizon)), x_final.reshape((13,1)))

                        mrp = self.quat_to_rodrig(x_final[3:7])
                        x_final_reduced = np.concatenate((x_final[0:3],mrp,x_final[7:10],x_final[10:13]), axis=None)  
                        self.des_state_reduced = np.kron(np.ones((1,self.N+self.MPC_Horizon)), x_final_reduced.reshape((12,1)))

                        inter_points = np.array(quat_way_points[1:-1])
                        reduced_inter_points = np.array(reduced_way_points[1:-1])
                                
                        for i in inter_times:
                                index = int((i/self.h))
                                way_point_index.append(index)

                        self.index_array = way_point_index.copy()
                        self.index_array.insert(0,0)

                        for i in range(len(self.index_array)-1):
                                self.des_state[:,self.index_array[i]:self.index_array[i+1]+1] = inter_points[i,:].reshape(13,1) 
                                self.des_state_reduced[:,self.index_array[i]:self.index_array[i+1]+1] = reduced_inter_points[i,:].reshape(12,1)
                
        def PID_AutoTune(self,states,scaling_factor):
                self.des_state =  np.array([3.0,3,3, 0,0,0, 0,0,0, 0,0,0])
                self.init_state_tune =  np.array([3.0,3,3, 0,0,0, 0,0,0, 0,0,0])

                actual_sim_time = self.tf
                actual_time_step = self.h
                actual_N = self.N

                self.PID_Gains =  np.zeros((6,3)) 
       
                # for i in [3,4,5,2,0,1]:  
                for i in states:  
                        if i < 3:
                                self.tf = 2
                                self.h =0.05
                                self.N = int(self.tf/self.h)+1  

                                inner_loop_rate = 1/self.h
                                outer_loop_rate = inner_loop_rate/4
                                self.outer_loop_h = 1/outer_loop_rate
                                self.t_outer = self.outer_loop_h   
                        else:
                                self.tf = 1
                                self.h =0.05
                                self.N = int(self.tf/self.h)+1       

                        self.PID_tune_state(i,scaling_factor)

                print(self.PID_Gains)
     
                self.tf = actual_sim_time
                self.h = actual_time_step   
                self.N = actual_N          
 
        def PID_tune_state(self,idx,scaling_factor):
                
                print(f"Started tuning for state {idx+1}")
                print(f" ")
                self.des_state = np.copy(self.init_state_tune)
                
                if idx > 2:
                        # r = R.from_euler('zyx', [90, 45, 30], degrees=True)
                        # quat = R.from_quat([self.init_state_tune[4,i], self.init_state_tune[5,i], self.init_state_tune[6,i], self.init_state_tune[3,i]])
                        # eul = quat.as_euler('xyz', degrees=True)                        
                        self.des_state[idx] += 5
                else:
                        self.des_state[idx] += 0.5
                
                self.x_integral = 0      
                self.y_integral = 0                          
                self.z_integral = 0
                self.phi_integral = 0
                self.theta_integral = 0
                self.psi_integral = 0

                self.PID_Gains[idx,0] = 0.1*scaling_factor
                
                overshoot = 2
                ss_error = 2
                rise_time = 2

                prop_gain_factor = 0.2
                prop_iterations = 0
                der_gain_factor = 0.2
                der_iteraions = 0

                tuned_state_var = np.zeros(self.N)

                while np.linalg.norm([rise_time,ss_error,overshoot]) > 0.9:
                        print(f"Current termination Critera (reqd. <0.7): {np.linalg.norm([rise_time,ss_error,overshoot])}")
                        if idx > 2:
                                state, t = self.simulate(self.init_state_tune, self.PID_attitude_controller)
                        else:
                                state, t = self.simulate(self.init_state_tune, self.PID_position_controller)

                        if idx > 2:
                                for i in range(self.N):
                                        quat = R.from_quat([state[4,i], state[5,i], state[6,i], state[3,i]])
                                        eul = quat.as_euler('xyz', degrees=True)
                                        tuned_state_var[i] = eul[idx-3]
                        else:
                                tuned_state_var  = state[idx,:]

                        if max(tuned_state_var[0:int(self.N/2)]) >= self.des_state[idx]:

                                rise_time = 0

                                overshoot = max(tuned_state_var) - self.des_state[idx]
                                if overshoot > 0.2:
                                        self.PID_Gains[idx,2] += scaling_factor*(der_gain_factor + 0.2*(abs(overshoot)))
                                        der_iteraions += 1
                                        if der_iteraions >= 10:
                                                der_gain_factor += 0.1
                                                scaling_factor += 0.5
                                                der_iteraions = 0
                                                print(f"")
                                                print(f"Increased deriv gain factor to {der_gain_factor}")
                                                print(f"")                                        
                                else:
                                        ss_error = abs(self.des_state[idx]-tuned_state_var[-1])
                                        if ss_error > 0.1:
                                                self.PID_Gains[idx,1] += scaling_factor*(0.05 + 0.01*(ss_error))
                                        
                        else:
                                rise_time = abs((self.des_state[idx]-max(tuned_state_var[0:int(self.N/2)])))
                                self.PID_Gains[idx,0] += scaling_factor*(prop_gain_factor + 0.2*rise_time)
                                prop_iterations += 1
                                if prop_iterations >= 10:
                                        prop_gain_factor += 0.1
                                        scaling_factor += 0.5
                                        prop_iterations = 0
                                        print(f"")
                                        print(f"Increased prop gain factor to {prop_gain_factor}")
                                        print(f"")  

                                
                # return 0
                
        def PID_attitude_controller(self,state,t,i):
                u = ((1/self.Kf)*self.m*self.g*0.25)*np.ones([4,1])

                quat = R.from_quat([state[4], state[5], state[6], state[3]])
                eul = quat.as_euler('xyz', degrees=True)

                phi, phi_des = eul[0], self.des_state[3,i]
                ang_vel_x, ang_vel_x_des = state[10], self.des_state[9,i]    #using omega from state is an approximation
                theta, theta_des = eul[1], self.des_state[4,i]
                ang_vel_y, ang_vel_y_des = state[11], self.des_state[10,i]    #using omega from state is an approximation
                psi, psi_des = eul[2], self.des_state[5,i]
                ang_vel_z, ang_vel_z_des = state[12], self.des_state[11,i]    #using omega from state is an approximation

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
                u = u + x

                return u.squeeze()

        def PID_position_controller(self,state,t,i):

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
        
        def gen_traj(self,way_points,inter_times,Traj_Params):

                self.mode = "TT"
                #for plotting
                self.waypoints = way_points

                # ic_q = R.from_quat([xic[4],xic[5],xic[6],xic[3]])
                # goal_q = R.from_quat([xgoal[4],xgoal[5],xgoal[6],xgoal[3]])

                # ic_mrp = ic_q.as_mrp()
                # goal_mrp =goal_q.as_mrp()

                # xic = np.array([
                #       xic[0:3], ic_mrp, xic[7:10],xic[10:]
                # ], dtype =float).reshape(12)
                # xgoal = np.array([
                #       xgoal[0:3], goal_mrp, xgoal[7:10],xgoal[10:]
                # ], dtype =float).reshape(12)           

                # planner = DIRCOL(self.param_dict)
                # self.DIRCOL_control = planner.solve_NLP(xic,xgoal)  

                planner = iLQR(self.param_dict,self.quad_rk4_step,way_points,inter_times,Traj_Params)
                self.iLQR_control,self.K,self.xref = planner.calc_trajectory()  

        def PTP_lqr_controller(self,state,t,i):
                
                if i in self.index_array:
                        self.xref =  jnp.array(self.des_state[:,i+1], dtype =float)
                        self.uref = ((1/self.Kf)*self.m*self.g*0.25)*jnp.ones([4],dtype =float)
                        
                        A = jax.jacfwd(lambda y: self.quad_rk4_step(y, self.uref))(self.xref)
                        B = jax.jacfwd(lambda y: self.quad_rk4_step(self.xref,y))(self.uref)
                                                
                        A_mod = (self.E(self.xref[3:7]).T)@A@self.E(self.xref[3:7])
                        B_mod = (self.E(self.xref[3:7]).T)@B

                        self.K, S, E = ct.dlqr(A_mod, B_mod, self.Q_lqr, self.R_lqr) 


                q0 = self.xref[3:7]
                q = state[3:7]
                phi = self.quat_to_rodrig(self.L(q0).T@q)

                del_x = jnp.block([state[0:3]-self.xref[0:3],phi,state[7:10]-self.xref[7:10],state[10:13]-self.xref[10:13]]) 
                u = self.uref - self.K@del_x
                return u
        
        def Ilqr_controller(self,state,t,i):

                control_list = self.iLQR_control[:,i]
                self.uref =  jnp.array([control_list[0],control_list[1],control_list[2],control_list[3]]).squeeze()

                q0 = self.xref[3:7,i]
                q = state[3:7]
                phi = self.quat_to_rodrig(self.L(q0).T@q)

                del_x = jnp.block([state[0:3]-self.xref[0:3,i],phi,state[7:10]-self.xref[7:10,i],state[10:13]-self.xref[10:13,i]]) 
                u = self.uref - self.K[:,:,i]@del_x
                return u        

        def MPC_controller(self,state,t,index):

                if index in self.index_array:
                        self.xref =  jnp.array(self.des_state[:,index+1], dtype =float)
                        self.uref = ((1/self.Kf)*self.m*self.g*0.25)*jnp.ones([4],dtype =float)

                        mrp = self.quat_to_rodrig(self.xref[3:7])
                        self.xref_reduced = np.concatenate((self.xref[0:3],mrp,self.xref[7:10],self.xref[10:13]), axis=None)                 
                        
                        A = jax.jacfwd(lambda y: self.quad_rk4_step(y, self.uref))(self.xref)
                        B = jax.jacfwd(lambda y: self.quad_rk4_step(self.xref,y))(self.uref)
                                                
                        self.A_mrp = (self.E(self.xref[3:7]).T)@A@self.E(self.xref[3:7])
                        self.B_mrp = (self.E(self.xref[3:7]).T)@B 

                x_goal_horizon =  self.des_state_reduced[:,index:index+self.MPC_Horizon]
                del_x = cp.Variable(12*self.MPC_Horizon)
                del_u = cp.Variable(4*(self.MPC_Horizon-1))
                constraints = []

                cost = 0.0
                for i in range(self.MPC_Horizon):
                        xi = self.xref_reduced + del_x[i*12:(i*12)+12]
                        cost += 0.5*cp.quad_form(xi - x_goal_horizon[:,i], self.Q_MPC)

                for i in range(self.MPC_Horizon-1):
                        ui = self.uref  + del_u[i*4:(i*4)+4]
                        cost += 0.5*cp.quad_form(ui, self.R_MPC)

                obj = cp.Minimize(cost)
                mrp = self.quat_to_rodrig(state[3:7])
                x0 = np.concatenate((state[0:3],mrp,state[7:10],state[10:13]), axis=None) 

                # initial condition constraint
                constraints += [self.xref_reduced + del_x[0:12] == x0]

                # add dynamics constraints
                for i in range(self.MPC_Horizon-1):
                        constraints += [ del_x[(i+1)*12:((i+1)*12)+12] ==  (self.A_mrp@del_x[i*12:(i*12)+12]) + (self.B_mrp@del_u[i*4:(i*4)+4]) ]
                        for j in del_u[i*4:(i*4)+4]:
                                constraints +=  [j  >=  -((1/self.Kf)*self.m*self.g*0.25)]   
                                constraints +=  [j  <=  self.m*self.g*self.clip_factor]    

                prob = cp.Problem(obj, constraints)
                prob.solve()  # Returns the optimal value.   
                print("status:", prob.status)
                print("value:", prob.value)

                u_soln = del_u.value
                # print(f"{u_soln[0:4]} at {index}" )
                return self.uref + u_soln[0:4] 

        def L(self,q):
                #Takes a quaternion and returns a matrix for left multiplication
                s = q[0]
                v = jnp.array(q[1:]).reshape((3,1))
                v_t = v.T

                Lq = jnp.block([
                        [s,-v_t],
                        [v, (s*jnp.eye(3))+self.hat_operator(v)]
                ])
                return Lq

        def R(self,q):
                #Takes a quaternion and returns a matrix for right multiplication
                s = q[0]
                v = jnp.array(q[1:]).reshape((3,1))
                v_t = v.T

                Rq = jnp.block([
                        [s,-v_t],
                        [v, (s*jnp.eye(3))-self.hat_operator(v)]
                ])
                return Rq

        def quat_to_rotmat(self,q):
                # Converts quaternion to rotation matrix
                T = - jnp.eye(4)  
                T = T.at[0,0].set(1)  
                #T[0,0]=1            
                H = jnp.block([
                        [jnp.zeros([1,3])],
                        [jnp.eye(3)]
                ])
                # Q = (H.T)@self.L(q)@(self.R(q).T)@H
                Q = (H.T)@T@(self.L(q))@T@(self.L(q))@H
                return Q            

        def calc_attitude_jacobian(self,q):
                # Calculate attitude jacobian at q
                H = jnp.block([
                        [jnp.zeros([1,3])],
                        [jnp.eye(3)]
                ])
                G = self.L(q)@H
                return G

        def CBF(self,x,u):
                A = jax.jacfwd(lambda y: self.quad_rk4_step(y,u))(x)
                B = jax.jacfwd(lambda y: self.quad_rk4_step(x,y))(u)
                A = np.array(A)
                B = np.array(B)
                u_var = cp.Variable(4)
                coeff = -10
                constraints = []

                def phi1(state):
                        dist1 = state[0:3]-np.array(self.obs_coords[0])
                        return jnp.inner(dist1,dist1)-(0.5+self.l)**2
                
                def phi2(state):
                        dist2 = state[0:3]-np.array(self.obs_coords[1])
                        return jnp.inner(dist2,dist2)-(0.5+self.l)**2
                
                grad_phi1 = jax.grad(phi1)(x)
                grad_phi2 = jax.grad(phi2)(x)

                grad_phi1 = np.array(grad_phi1)
                grad_phi2 = np.array(grad_phi2)                

                # xdot = ((A@x + B@u_var.value) - x)/self.h
                # phi_dot1 = float(grad_phi1.T@xdot)
                # phi_dot2 = float(grad_phi2.T@xdot)

                obj = cp.Minimize(cp.sum_squares(u - u_var))
                
                # must be safe
                # constraints += [phi_dot1 <= coeff*float(phi1(x))]
                # constraints += [phi_dot2 <= coeff*float(phi2(x))]

                constraints += [grad_phi1.T@( ((A@x + B@u_var) - x)/self.h)  >= coeff*float(phi1(x))]
                constraints += [grad_phi2.T@( ((A@x + B@u_var) - x)/self.h)  >= coeff*float(phi2(x))]                     
                # constraints += [u_var  >= 0]                
                
                prob = cp.Problem(obj, constraints)
                prob.solve()  # Returns the optimal value.   
                print("status:", prob.status)
                if prob.status =='optimal': 
                        if not np.array_equal(u_var.value, u):
                                print("u modified to avoid obstacle")                            
                        return u_var.value  
                else:
                        return u             

        @staticmethod
        def rodrig_to_quat(phi):
                phi = np.array(phi).reshape((3,1))
                a = 1/np.sqrt(phi.T@phi)
                return a*np.block([ [1],[phi] ])

        @staticmethod
        def quat_to_rodrig(quat):
                quat = np.array(quat).reshape((4,))
                return quat[1:4]/quat[0]
        
        def E(self,q):
                return block_diag(np.eye(3), self.calc_attitude_jacobian(q), np.eye(6))
                
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
        
        @staticmethod
        def euler_from_quaternion(x, y, z, w):
                """
                Convert a quaternion into euler angles (roll, pitch, yaw)
                roll is rotation around x in radians (counterclockwise)
                pitch is rotation around y in radians (counterclockwise)
                yaw is rotation around z in radians (counterclockwise)
                """
                t0 = +2.0 * (w * x + y * z)
                t1 = +1.0 - 2.0 * (x * x + y * y)
                roll_x = math.atan2(t0, t1)
        
                t2 = +2.0 * (w * y - z * x)
                t2 = +1.0 if t2 > +1.0 else t2
                t2 = -1.0 if t2 < -1.0 else t2
                pitch_y = math.asin(t2)
        
                t3 = +2.0 * (w * z + x * y)
                t4 = +1.0 - 2.0 * (y * y + z * z)
                yaw_z = math.atan2(t3, t4)
        
                return roll_x, pitch_y, yaw_z # in radians