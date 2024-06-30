import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import block_diag
from jax import jit
from scipy.spatial.transform import Rotation as R

class iLQR:
        # Constructor
        def __init__(self, param_dict, dynfn,way_points,Traj_Params) -> None:
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
                self.dynfn = jit(dynfn)
                # self.T = - np.eye(4)   
                # self.T[0,0]=1  
                # self.H = np.block([
                #         [np.zeros([1,3])],
                #         [np.eye(3)]
                # ])
                # self.qtoQ_inter = (self.H.T)@self.T
                self.nx = 12
                self.nu = 4
                self.N = int(self.tf/self.h)+1
                self.t_vec = np.linspace(0, self.tf, num=self.N)          

                self.Q  = Traj_Params[0]
                self.R  = Traj_Params[1]
                self.Qf = Traj_Params[2]  

                quat_way_points = []

                for i in way_points:
                        r = R.from_euler('xyz', [i[3], i[4], i[5]], degrees=True)
                        quat = r.as_quat()
                        quat_way_points.append(np.concatenate((i[0:3],[quat[3],quat[0],quat[1],quat[2]],i[6:9],i[9:]), axis=None))

                self.xic = quat_way_points[0]
                mrp_way_points = []

                for i in quat_way_points:
                        mrp = self.quat_to_rodrig(i[3:7])
                        mrp_way_points.append( np.concatenate((i[0:3],mrp,i[7:10],i[10:13]), axis=None) )

                inter_points = mrp_way_points[1:-1]
                x_final =  jnp.array(mrp_way_points[-1], dtype =float)
                self.xgoal = np.kron(np.ones((1,self.N)), x_final.reshape((12,1)))
                xgoal_step = int(np.floor((self.N-1)/len(mrp_way_points)))

                for count, point in enumerate(inter_points):
                        self.xgoal[:,xgoal_step*count:(xgoal_step*count)+xgoal_step] = np.kron(np.ones((1,xgoal_step)), np.array(point).reshape((12,1)))

        def calc_jacobians(self,xref,xref_2,uref):
                A = jax.jacfwd(lambda y: self.dynfn(y, uref))(xref)
                B = jax.jacfwd(lambda y: self.dynfn(xref,y))(uref)    

                A_mod = (self.E(xref_2[3:7]).T)@A@self.E(xref[3:7])
                B_mod = (self.E(xref_2[3:7]).T)@B                            
                
                return A_mod,B_mod
        
        def E(self,q):
                return block_diag(np.eye(3), self.calc_attitude_jacobian(q), np.eye(6))

        def calc_attitude_jacobian(self,q):
                # Calculate attitude jacobian at q
                H = jnp.block([
                        [jnp.zeros([1,3])],
                        [jnp.eye(3)]
                ])
                G = self.L(q)@H
                return G       
                 
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

        def calc_cost(self,xtraj,utraj):
                J = 0
                for i in range(self.N-1):
                        x = xtraj[:,i]-self.xgoal[:,i]
                        u = utraj[:,i]
                        J += 0.5*((x.T)@self.Q@x)+ 0.5*(u.T@self.R@u)
                x = xtraj[:,self.N-1]-self.xgoal[:,self.N-1]                    
                J += 0.5*((x.T)@self.Qf@x)
                return float(J)
        
        def rodrig_to_quat(self,phi):
                phi = np.array(phi).reshape((3,1))
                a = 1/np.sqrt(phi.T@phi)
                return a*np.block([ [1],[phi] ])

        def quat_to_rodrig(self,quat):
                quat = np.array(quat).reshape((4,))
                return quat[1:4]/quat[0]
        
        def backward_pass(self,p,P,d,K,xtraj,xtraj_reduced,utraj):
                print("Starting Backward Pass")
                delta_J = 0.0
                for k in range(self.N-2,-1,-1):
                        if (k%50==0):
                                print(f"{k} out of {self.N-2}")
                        #Calculate derivatives
                        q = self.Q@(xtraj_reduced[:,k]-self.xgoal[:,k])
                        r = self.R@utraj[:,k]
                
                        A,B = self.calc_jacobians(xtraj[:,k],xtraj[:,k+1], utraj[:,k]) #ForwardDiff.jacobian(dx->dynamics_rk4(dx,utraj[k]),xtraj[:,k])
                
                        gx = q + A.T@p[:,k+1]
                        gu = r + B.T@p[:,k+1]
                        
                        #iLQR (Gauss-Newton) version
                        Gxx = self.Q + A.T@P[:,:,k+1]@A
                        Guu = self.R + B.T@P[:,:,k+1]@B
                        Gxu = A.T@P[:,:,k+1]@B
                        Gux = B.T@P[:,:,k+1]@A
                        
                        β = 0.1
                        while True:
                                regul_matrix = np.block([[Gxx,Gxu],[Gux, Guu]])
                                if np.all(np.linalg.eigvals(regul_matrix) > 0):
                                        break
                                else:
                                        I =np.eye(12)
                                        Gxx += A.T@β@I@A
                                        Guu += B.T@β@I@B
                                        Gxu += A.T@β@I@B
                                        Gux += B.T@β@I@A
                                        β = 2*β
                                        print("regularizing G")
                        d[:,k] = np.linalg.solve(Guu, gu) 
                        K[:,:,k] = np.linalg.solve(Guu, Gux)
                
                        p[:,k] = gx - K[:,:,k].T@gu + K[:,:,k].T@Guu@d[:,k] - Gxu@d[:,k]
                        P[:,:,k] = Gxx + K[:,:,k].T@Guu@K[:,:,k] - Gxu@K[:,:,k] - K[:,:,k].T@Gux
                
                        delta_J += gu.T@d[:,k]
                
                return float(delta_J)       

        def calc_trajectory(self):
                print("Starting Trajectory Calculation")
                xic =  jnp.array(self.xic, dtype =float)
                uhover = ((1/self.Kf)*self.mass*self.g*0.25)*jnp.ones([4],dtype =float)
                xtraj = np.kron(np.ones((1,self.N)), xic.reshape((13,1)))
                utraj = np.kron(np.ones((1,self.N-1)), uhover.reshape((4,1)))

                xtraj_reduced = np.array(xtraj[0:12,:])
                mrp = self.quat_to_rodrig(xtraj[3:7,0])
                xtraj_reduced[:,0] = np.concatenate((xtraj[0:3,0],mrp,xtraj[7:10,0],xtraj[10:13,0]), axis=None)                 
 

                #Initial Rollout
                for i in range(self.N-1):
                        xtraj[:,i+1] = self.dynfn(xtraj[:,i],utraj[:,i])
                        mrp = self.quat_to_rodrig(xtraj[3:7,i+1])
                        xtraj_reduced[:,i+1] = np.concatenate((xtraj[0:3,i+1], mrp,xtraj[7:10,i+1],xtraj[10:13,i+1]), axis=None)               

                print("Initial Rollout Done")

                J = self.calc_cost(xtraj_reduced,utraj)  

                #iLQR Algorithm
                p = np.ones((self.nx,self.N))
                P = np.zeros((self.nx,self.nx,self.N))
                d = np.ones((self.nu,self.N-1))
                K = np.zeros((self.nu,self.nx,self.N-1))
                delta_J = 1.0

                xn = np.zeros((self.nx,self.N))
                un = np.zeros((self.nu,self.N-1))

                gx = np.zeros(self.nx)
                gu = np.zeros(self.nu)
                Gxx = np.zeros((self.nx,self.nx))
                Guu = np.zeros((self.nu,self.nu))
                Gxu = np.zeros((self.nx,self.nu))
                Gux = np.zeros((self.nu,self.nx))

                print("Starting iLQR loop")
                iterations = 0
                max_prev =1000

                while (np.max(abs(d)) > 0.0005) and ( abs( max_prev-(np.max(abs(d))) )>0.000001):
                        max_prev = np.max(abs(d))
                        p[:,self.N-1] = self.Qf@(xtraj_reduced[:,self.N-1]-self.xgoal[:,self.N-1])
                        P[:,:,self.N-1] = self.Qf                         
                        delta_J = self.backward_pass(p,P,d,K,xtraj,xtraj_reduced,utraj)
                        print("Backward Pass Done")

                        #Forward rollout with line search
                        xn = np.array(xtraj)
                        un = np.array(utraj)
                        xn_reduced = np.array(xtraj_reduced)
                        alpha = 1.0

                        for k in range(self.N-1):
                                un[:,k] = utraj[:,k] - (alpha*d[:,k]) - K[:,:,k]@(xn_reduced[:,k]-xtraj_reduced[:,k])
                                xn[:,k+1] = self.dynfn(xn[:,k],un[:,k])
                                mrp = self.quat_to_rodrig(xn[3:7,k+1])
                                xn_reduced[:,k+1] = np.concatenate((xn[0:3,k+1], mrp,xn[7:10,k+1],xn[10:13,k+1]), axis=None) 

                        Jn = self.calc_cost(xn_reduced,un) 
                        
                        # while isnan(Jn) || Jn > (J - 1e-2*alpha*ΔJ)
                        while Jn > (J - 1e-2*alpha*delta_J):
                                alpha = 0.5*alpha
                                print(f"Reducing alpha to {alpha}")
                                for k in range(self.N-1):
                                        un[:,k] = utraj[:,k] - (alpha*d[:,k]) - K[:,:,k]@(xn_reduced[:,k]-xtraj_reduced[:,k])
                                        xn[:,k+1] = self.dynfn(xn[:,k],un[:,k])
                                        mrp = self.quat_to_rodrig(xn[3:7,k+1])
                                        xn_reduced[:,k+1] = np.concatenate((xn[0:3,k+1], mrp,xn[7:10,k+1],xn[10:13,k+1]), axis=None) 

                                Jn = self.calc_cost(xn_reduced,un) 
                        
                        iterations += 1
                        print(f"Forward Pass Done, iteration: {iterations}")
                        print(f"Current tolerance: {np.max(abs(d))}, Required: <0.0005 ")
                        print(f"Other terminataion critera:{abs( max_prev-(np.max(abs(d))) )},Required: <0.001")
                        J = Jn
                        xtraj = np.array(xn)
                        xtraj_reduced = np.array(xn_reduced)
                        utraj = np.array(un)

                return utraj,K,xtraj
