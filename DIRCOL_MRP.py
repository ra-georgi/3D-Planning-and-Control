import numpy as np
from gekko import GEKKO

class DIRCOL:
        
        # Constructor
        def __init__(self, param_dict) -> None:
                self.mass = param_dict["Mass"]
                self.Ixx = param_dict["Ixx"]
                self.Iyy = param_dict["Iyy"]
                self.Izz = param_dict["Izz"]
                self.l = param_dict["ArmLength"]
                self.Kf = param_dict["Kf"]
                self.Km = param_dict["Km"]
                self.g = param_dict["Accel_g"]
                self.h = param_dict["TimeStep"]
                self.tf = param_dict["FinalTime"]
                self.T = - np.eye(4)   
                self.T[0,0]=1  
                # self.H = np.block([
                #         [np.zeros([1,3])],
                #         [np.eye(3)]
                # ])
                # self.qtoQ_inter = (self.H.T)@self.T

        def quad_dynamics(self,x,u):
                # Takes in current state and returns xdot
                # norm_var = np.linalg.norm(x[3:7])

                pos = x[0:3]
                mrp = x[3:6]
                vel = x[6:9]
                omega = x[9:]
                
                Q = self.dcm_from_mrp(mrp)
                
                m = self.mass 
                Kf = self.Kf
                Km = self.Km
                l = self.l
                g = self.g

                I = np.array([[self.Ixx,0,0],[0,self.Iyy,0],[0,0,self.Izz]])

                F1 = Kf*u[0]
                F2 = Kf*u[1]
                F3 = Kf*u[2]
                F4 = Kf*u[3]

                F = np.array([0, 0, F1+F2+F3+F4]) #total rotor force in body frame
                f =  np.array([0,0,-m*g])+ Q@F

                M1 = Km*u[0]
                M2 = Km*u[1]
                M3 = Km*u[2]
                M4 = Km*u[3]                

                tau_b = np.array([l*(F2-F4),l*(F3-F1),(M1-M2+M3-M4)])

                vel_dot = f/m

                const = mrp[0]**2+mrp[1]**2+mrp[2]**2 
                temp_inter = self.hat_operator(mrp)

                mrp_dot = (   I + 2*((temp_inter@temp_inter) + temp_inter)/(1+const)   )@omega
                # mrp_dot =((1+const)/4)*mrp_dot
                omega_dot= np.linalg.inv(I)@(tau_b - (self.hat_operator(omega)@I@omega))
     
                return [vel[0],vel[1],vel[2],
                        ((1+const)/4)*mrp_dot[0], ((1+const)/4)*mrp_dot[1],((1+const)/4)*mrp_dot[2],
                        vel_dot[0],vel_dot[1],vel_dot[2],
                        omega_dot[0],omega_dot[1],omega_dot[2]]      
                # return np.block([
                #         pos_dot,quat_dot,vel_dot,omega_dot
                # ])        
        
        @staticmethod
        def dcm_from_mrp(p):
                p1 = p[0]
                p2 = p[1]
                p3 = p[2]

                den = (p1**2 + p2**2 + p3**2 + 1)**2
                a = (4*p1**2 + 4*p2**2 + 4*p3**2 - 4)

                Q = np.array([
                       [(-((8*p2**2+8*p3**2)/den-1)*den),     (8*p1*p2 + p3*a),       (8*p1*p3 - p2*a)], 
                       [(8*p1*p2 - p3*a),           (-((8*p1**2 + 8*p3**2)/den - 1)*den),        (8*p2*p3 + p1*a)],
                       [(8*p1*p3 + p2*a),            (8*p2*p3 - p1*a),                   (-((8*p1**2 + 8*p2**2)/den - 1)*den)]
                       ])
                
                # Q = np.zeros([3,3])

                # Q[0,0] = (-((8*p2**2+8*p3**2)/den-1)*den)
                # Q[0,1] = (8*p1*p2 + p3*a) 
                # Q[0,2] = (8*p1*p3 - p2*a)

                # Q[1,0] = (8*p1*p2 - p3*a)
                # Q[1,1] = (-((8*p1**2 + 8*p3**2)/den - 1)*den)
                # Q[1,2] = (8*p2*p3 + p1*a)

                # Q[2,0] = (8*p1*p3 + p2*a)
                # Q[2,1] = (8*p2*p3 - p1*a)
                # Q[2,2] = (-((8*p1**2 + 8*p2**2)/den - 1)*den)

                Q = Q/den
        
                return Q
      

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

        def solve_NLP(self,xic,xgoal):
                
                nx = 12
                nu = 4
                N = int(self.tf/self.h)+1
                t_vec = np.linspace(0, self.tf, num=N)          

                Q = np.diag(np.ones(nx))
                R = 0.1*np.diag(np.ones(nu))
                Qf = 10*np.diag(np.ones(nx))    
  
                #Initialize Model
                self.m = GEKKO(remote=False)
                # self.m.options.MAX_ITER=250
                #Set global options
                self.m.options.IMODE = 3 #steady state optimization   
                self.m.options.SOLVER = 3
                # self.m.options.RTOL=1e-2
                # self.m.options.OTOL=1e-2              
                state = []
                control = []
                # self.m.options.REDUCE=2
                dt_by6 = self.m.Const(self.h/6)     
                dt_by8 = self.m.Const(self.h/8)                
                for i in np.arange(0,N):
                       state.append(self.m.Array(self.m.Var,nx,value=0,lb=-1e5,ub=1e5))
                       control.append(self.m.Array(self.m.Var,nu,value=5*(self.mass*self.g),lb=0,ub=5*(self.mass*self.g)))
                       if i == 0:
                              self.m.Minimize(0.5*(state[i]-xgoal).T@Q@(state[i]-xgoal) + 0.5*control[i].T@R@control[i])
                              for j in range(nx):
                                     IC = state[0]     
                                     self.m.Equation(IC[j]==xic[j])                        
                       elif i == N-1:
                              self.m.Minimize(0.5*(state[i]-xgoal).T@Qf@(state[i]-xgoal))
                              for j in range(nx):
                                     FC = state[N-1]     
                                     self.m.Equation(FC[j]==xgoal[j])                               
                       else: 
                            self.m.Minimize(0.5*(state[i]-xgoal).T@Q@(state[i]-xgoal) + 0.5*control[i].T@R@control[i])


                # Hermite Simpson 
                # for i in np.arange(0,N-1):      
                #         x_dot_k = np.array(self.quad_dynamics(state[i],control[i]))
                #         x_dot_kplus1 = np.array(self.quad_dynamics(state[i+1],control[i]))
                #         x_inter = 0.5*(state[i]+state[i+1]) + dt_by8*(x_dot_k-x_dot_kplus1)
                #         x_dot_inter = np.array(self.quad_dynamics(x_inter,control[i]))
                #         x_residual = state[i]+dt_by6*(x_dot_k+(4*x_dot_inter)+x_dot_kplus1)-state[i+1]
                #         for i in x_residual:
                #                 self.m.Equation(i==0)  

                                # temp = state[1]
                                # self.m.Equation(x_residual[i]==temp[i]) 

                ## Forward Euler
                for i in np.arange(0,N-1):      
                        temp = state[i+1]-state[i]-self.h*np.array(self.quad_dynamics(state[i], control[i]))
                        for i in temp:
                                self.m.Equation(i==0)    

  
                self.m.open_folder()  
                print("Starting solving NLP")
                self.m.solve(disp=True)
                print("NLP Solved")
                return control

                # m.cleanup()





        
