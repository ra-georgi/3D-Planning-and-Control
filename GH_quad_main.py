from GH_Quadcopter import Quadcopter
from default_quad_params import param_dict
import numpy as np
from scipy.linalg import block_diag

#Initalize Quadcopter Object with required parameters
quad1 = Quadcopter(param_dict)

# Additional Features for simulation can be provided through the sim_params variable in the format (feature present?, related params)
# For collision avoidance, first tuple represents (CA mode, CBF on, obstacles coordinates)
# For control constraints, the variable is a clipping factor
# obstacles = ([9,9,9],[10,10,10])
obstacles = ([4,4,4],[5.5,5.5,5]) #Use comma if only one obstacle
# obstacles = ()
clip_factor = 10
sim_params =(True,(True,obstacles),True,clip_factor)
quad1.setup_sim(sim_params)

# 2(3) simulation types: Point To Point (PTP) and Trajectory Tracking (TT)

# PTP can be run using Controllers based on PID or LQR 
# Specify: -Starting and Goal States - Controller Gains for PID - Q and R matrices for LQR 

#Provide states in this format : [position,euler_angles,velocity,angular velocity]
# (Euler angles are more intuitive for the purpose of quickly testing out different rotations)


x0    = [3.0,3,3, 0,0,0, 0,0,0, 0,0,0]
xgoal = [7.0,7,5, 0,0,0, 0,0,0, 0,0,0]

# xinter_1 = [4.0,3,3, 0,0,0, 0,0,0, 0,0,0]
# inter_times = [2]
# way_points = (x0,xinter_1,xgoal)

inter_times = []
way_points = (x0,xgoal)


# *********************************************************************************

#PID Parameters

#Quadcopter Object for PID tuning with required sim params
# quad2 = Quadcopter(param_dict)

# #Either Autotune or setup manually
# PID_Gains =[]
# PID_Gains = [[2.56278919e+01, 0., 2.28994728e+01],
#              [2.56278919e+01, 0., 2.28994728e+01],
#              [2.57012684e+00, 0,  9.94091206e-01],
#              [3.74266832e-03, 0.,         3.20275599e-02 ],
#              [3.74266832e-03, 0.,         3.20275599e-02],
#              [4.98906794e-03, 0.,         5.01777145e-02]]

# quad1.PID_Gains[3:,:] = np.array(PID_Gains)[3:,:]

# states = [3,4,5,2,0,1]
# scaling_factor = 0.01
# quad1.PID_AutoTune(states,scaling_factor)
                      
# PID_Gains  = [[ 12,   0.5,  10],  # Kp_x, Ki_x, Kd_x 
#               [ 12,   0.5,  10],  # y
#               [  4,   0.5,   2],  # z
#               [0.5,  0.05,   4],  # phi
#               [0.5,  0.05,   4],  # theta
#               [  0,     0,   0]]  # psi
 
# PTP_Params = np.array(PID_Gains)
# # quad1.setup_PTP(xgoal,PTP_Params,"PID")
# quad1.setup_PTP(way_points,inter_times,PTP_Params,"PID")
# state, t = quad1.simulate(x0, quad1.PID_position_controller)

# *********************************************************************************
# # LQR Parameters
# Q = 1*np.eye(12)
# R = 0.01*np.eye(4)
# PTP_Params = (Q,R)
# quad1.setup_PTP(way_points,inter_times,PTP_Params,"LQR")
# state, t = quad1.simulate(x0, quad1.PTP_lqr_controller)

# *********************************************************************************

# TT can use a combination of:
# Trajectory generation Techniques Available: 1) LQR 2) iLQR 3) DIRCOL 4)
# Tracking Techniques Available: 1) PID  2) LQR based Tracking controller 



# Q = 1*block_diag(np.eye(6), 0.1*np.eye(6))
# R = 0.1*np.diag(np.ones(4))
# Qf = 1*block_diag(np.eye(6), 0.1*np.eye(6))
# Q_wp = 50*block_diag(np.eye(6), 0.1*np.eye(6))

# Traj_Params = (Q,R,Qf,clip_factor,obstacles,Q_wp)

# quad1.gen_traj(way_points,inter_times,Traj_Params)
# state, t = quad1.simulate(x0, quad1.Ilqr_controller)

# *********************************************************************************
# MPC
Q = 1000*np.diag(np.ones(12))
R = 0.01*np.diag(np.ones(4))
Qf = 0.1*np.diag(np.ones(12))   
mpc_horizon = int(quad1.tf/(3*quad1.h))
# xgoal_traj

x_min = -1e3*np.ones(12)
x_max = 1e3*np.ones(12)
PTP_Params = (Q,R,Qf,mpc_horizon,x_min,x_max)   #Can put xgoal instead of xgoal_traj for PTP

# quad1.setup_MPC(MPC_Params)
quad1.setup_PTP(way_points,inter_times,PTP_Params,"MPC")
state, t = quad1.simulate(x0, quad1.MPC_controller)
# *********************************************************************************

quad1.quad_plots(state,t)
quad1.quad_animation(state,t)

