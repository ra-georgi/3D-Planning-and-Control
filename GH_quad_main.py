from GH_Quadcopter import Quadcopter
from default_quad_params import param_dict
import numpy as np

#Initalize Quadcopter Object with required parameters
quad1 = Quadcopter(param_dict)

# Additional Features for simulation can be provided through the sim_params variable in the format (feature present?, related params)
# For collision avoidance, first tuple represents (CA mode, CBF on, obstacles coordinates)
# For control constraints, the variable is a clipping factor
obstacles = ([3,3,3],[5,5,5])
clip_factor = 5
sim_params =(True,(False,obstacles),True,clip_factor)
quad1.setup_sim(sim_params)

# 2(3) simulation types: Point To Point (PTP) and Trajectory Tracking (TT)

# PTP can be run using Controllers based on PID or LQR 
# Specify: -Starting and Goal States - Controller Gains for PID - Q and R matrices for LQR 

#Provide states in this format : [position,euler_angles,velocity,angular velocity]
# (Euler angles are more intuitive for the purpose of quickly testing out different rotations)

x0    = [1,1,1, 0,0,0, 0,0,0, 0,0,0]
xgoal = [6.0,6,5, 0,0,0, 0,0,0, 0,0,0]
# xgoal = [7.0,7,5, 0,0,0, 0,0,0, 0,0,0]

# *********************************************************************************

#PID Parameters

#Quadcopter Object for PID tuning with required sim params
# quad2 = Quadcopter(param_dict)

#Either Autotune or setup manually

# PID_Gains = quad1.PID_AutoTune()
# print(PID_Gains)
# PID_position_gains = PID_Gains[0:9]
# PID_attitude_gains = PID_Gains[9:]

# #                      Kp_x,Ki_x,Kd_x  Kp_y, Ki_y, Kd_y   Kp_z, Ki_z, Kd_z
# PID_position_gains  = [12, 0.5, 10,    12, 0.5, 10,       4, 0.5, 2]
# #                      phi:Kp,Ki,Kd    theta:Kp,Ki,Kd     psi:Kp,Ki,Kd
# PID_attitude_gains  = [0.5, 0.05, 4,   0.5, 0.05, 4,      0, 0, 0]    

# #                      Kp_x,Ki_x,Kd_x  Kp_y, Ki_y, Kd_y   Kp_z, Ki_z, Kd_z
# PID_position_gains  = [8.27571298,  3.85741389,  5.12538807,  2.25150778,  0.97868183,  0.56570418,  4.11601854,  7.47453166,  2.01581331]
# #                      phi:Kp,Ki,Kd    theta:Kp,Ki,Kd     psi:Kp,Ki,Kd
# PID_attitude_gains  = [ 1.38780541,  4.44112354,  3.97322802,-1.35917063,  8.46071173,  7.72607081,  5.80069485,  1.42218637,  3.13017961]


# PTP_Params = (PID_position_gains,PID_attitude_gains)
# quad1.setup_PTP(xgoal,PTP_Params,"PID")
# state, t = quad1.simulate(x0, quad1.PID_position_controller)

# *********************************************************************************
#LQR Parameters
# Q = np.eye(12)
# R = 0.1*np.eye(4)
# PTP_Params = (Q,R)
# quad1.setup_PTP(xgoal,PTP_Params,"LQR")
# state, t = quad1.simulate(x0, quad1.PTP_lqr_controller)

# *********************************************************************************

# TT can use a combination of:
# Trajectory generation Techniques Available: 1) LQR 2) iLQR 3) DIRCOL 4)
# Tracking Techniques Available: 1) PID  2) LQR based Tracking controller 

xinter_1 = [3,5,1, 0,0,0, 1,1,0, 0,0,0]
xinter_2 = [5,7,1, 0,0,0, 1,1,0, 0,0,0]
# xinter_3 = [7,5,1, 0,0,0, 0,0,0, 0,0,0]
# way_points = (x0,xinter_1,xinter_2,xgoal)
way_points = (x0,xgoal)

Q = 0.1*np.diag(np.ones(12))
R = 0.1*np.diag(np.ones(4))
Qf = 10*np.diag(np.ones(12))   
Traj_Params = (Q,R,Qf,clip_factor,obstacles)

quad1.gen_traj(way_points,Traj_Params)
state, t = quad1.simulate(x0, quad1.Ilqr_controller)

# *********************************************************************************
# MPC
# Q = 10000*np.diag(np.ones(12))
# R = 0.1*np.diag(np.ones(4))
# Qf = 1*np.diag(np.ones(12))   
# mpc_horizon = int(quad1.tf/(4*quad1.h))
# # xgoal_traj
# times = np.linspace(0,4*np.pi,quad1.N)
# xgoal_traj = np.zeros([12,quad1.N])
# radius = 2
# center =[4,4]
# x0    = [center[0]+radius,center[1],4, 0,0,0, 0,0,0, 0,0,0]
# for index, time in enumerate(times):
#     xgoal_traj[0:3,index] = [center[0]+radius*np.cos(time),center[1]+radius*np.sin(time),4]

# for index, time in enumerate(times):
#     if (index > 0) and (time != times[-1]):
#         xgoal_traj[6:9,index] = (xgoal_traj[0:3,index+1] - xgoal_traj[0:3,index])/quad1.h


# x_min = -1e3*np.ones(12)
# x_max = 1e3*np.ones(12)
# MPC_clip_factor = clip_factor
# MPC_Params = (Q,R,Qf,xgoal_traj,mpc_horizon,x_min,x_max,MPC_clip_factor)   #Can put xgoal instead of xgoal_traj for PTP

# quad1.setup_MPC(MPC_Params)
# state, t = quad1.simulate(x0, quad1.MPC_controller)
# *********************************************************************************

quad1.quad_plots(state,t)
quad1.quad_animation(state,t)

