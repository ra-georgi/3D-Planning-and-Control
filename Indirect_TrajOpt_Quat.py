import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import block_diag
from jax import jit
from scipy.spatial.transform import Rotation as R

class iLQR:
        # Constructor
        def __init__(self, param_dict, dynfn, way_points,inter_times, Traj_Params) -> None:
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
                # self.Qf = block_diag(np.eye(3), np.zeros((4,4)), np.eye(6)) 
                self.clip_factor  = Traj_Params[3]
                self.obstacles = np.array(Traj_Params[4])  
                self.Q_wp = Traj_Params[5]

                # self.n_obs = np.shape(self.obstacles)[0]
                self.n_obs = len(Traj_Params[4])

                self.lg_mul = np.zeros((8+self.n_obs,self.N))
                self.I_penalty = np.zeros((8+self.n_obs,8+self.n_obs,self.N))
                self.penalty = 0.1
                self.scale_penalty = 10

                self.u_constraints_matrix = np.concatenate((np.eye(4), -np.eye(4)), axis=0)
                u_max = (self.mass*self.g*self.clip_factor)*np.ones((4,1))
                self.u_constraints_limits = np.concatenate((u_max, np.zeros((4,1))), axis=0)
                self.constraint_jac_u = np.zeros((8+self.n_obs,4))
                self.constraint_jac_u[0:8,:] = self.u_constraints_matrix

                self.state_to_pos = np.zeros((3,12))
                self.state_to_pos[0,0] = 1 
                self.state_to_pos[1,1] = 1 
                self.state_to_pos[2,2] = 1                 

                quat_way_points = []
                self.way_point_index = []
                self.scale_quat_penalty = 10

                if isinstance(way_points, tuple):

                        for i in way_points:
                                r = R.from_euler('xyz', [i[3], i[4], i[5]], degrees=True)
                                quat = r.as_quat()
                                quat_way_points.append(np.concatenate((i[0:3],[quat[3],quat[0],quat[1],quat[2]],i[6:9],i[9:]), axis=None))

                        self.xic = quat_way_points[0]
                        
                        # xgoal_step = int(np.floor((self.N-1)/ (len(quat_way_points)-1) ))
                        # for count, point in enumerate(inter_points):
                        #         self.xgoal[:,xgoal_step*count:(xgoal_step*count)+xgoal_step] = np.kron(np.ones((1,xgoal_step)), np.array(point).reshape((13,1)))

                        x_final =  jnp.array(quat_way_points[-1], dtype =float)
                        self.xgoal = np.kron(np.ones((1,self.N)), x_final.reshape((13,1)))
                        inter_points = np.array(quat_way_points[1:-1])
                                
                        for i in inter_times:
                                index = int((i/self.h))
                                self.way_point_index.append(index)

                        index_array = self.way_point_index.copy()
                        index_array.insert(0,0)

                        for i in range(len(index_array)-1):
                                self.xgoal[:,index_array[i]:index_array[i+1]+1] = inter_points[i,:].reshape(13,1)

                else:
                        quat_way_points = np.zeros((13,self.N))  
                        for i in range(self.N):
                                r = R.from_euler('xyz', [way_points[3,i], way_points[4,i], way_points[5,i]], degrees=True)
                                quat = r.as_quat()
                                quat_way_points[0:3,i] = way_points[0:3,i]
                                quat_way_points[3:7,i] = [quat[3],quat[0],quat[1],quat[2]]
                                quat_way_points[7:10,i] = way_points[6:9,i]
                                quat_way_points[10:,i] = way_points[9:,i]

                        self.xic = quat_way_points[:,0]
                        self.xgoal = quat_way_points

                self.adjust_dim_matrix = block_diag(np.eye(3), np.zeros((3,4)), np.eye(6)) 
                self.adjust_dim_matrix_quat = np.concatenate(( np.zeros((3,3)),np.eye(3),np.zeros((3,3)),np.zeros((3,3)) ), axis=0)
  

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
                #Start and Intermediate points
                for i in range(self.N-1):

                        # Penalizing waypoint errors more heavily than other intermediate points
                        if i in self.way_point_index:
                                Q = self.Q_wp
                                quat_penalty = self.scale_quat_penalty*self.Q_wp[0,0]  # To scale appropriately
                        else:
                                Q = self.Q
                                quat_penalty = self.scale_quat_penalty*self.Q[0,0]

                        x = self.adjust_dim_matrix@(xtraj[:,i]-self.xgoal[:,i])
                        u = utraj[:,i]-self.uhover

                        quat_current = xtraj[3:7,i]
                        quat_goal = self.xgoal[3:7,i]
                        J_quat = 1-np.abs((quat_goal.T)@quat_current)

                        J += 0.5*((x.T)@Q@x) + 0.5*(u.T@self.R@u) + (quat_penalty*J_quat)

                        constraints = self.constraint_inter(xtraj[:,i],utraj[:,i]).squeeze()
                        J += (self.lg_mul[:,i].T)@constraints
                        for j in range(8+self.n_obs):
                                if not ( (self.lg_mul[j,i] == 0) and (constraints[j]<=0) ):
                                        self.I_penalty[j,j,i] = self.penalty
                        J += (constraints.T)@self.I_penalty[:,:,i]@constraints

                #Final Position        
                x = self.adjust_dim_matrix@(xtraj[:,self.N-1]-self.xgoal[:,self.N-1]) 
                quat_current = xtraj[3:7,self.N-1]
                quat_goal = self.xgoal[3:7,self.N-1]
                J_quat = 1-np.abs((quat_goal.T)@quat_current)   

                quat_penalty = self.Qf[0,0]*self.scale_quat_penalty                              

                J += 0.5*((x.T)@self.Qf@x) + (quat_penalty*J_quat)

                constraints = self.constraint_final(xtraj[:,self.N-1]).squeeze()
                J += (self.lg_mul[:,self.N-1].T)@constraints
                for j in range(8,8+self.n_obs):
                        if not ( (self.lg_mul[j,self.N-1] == 0) and (constraints[j]<0) ):
                                self.I_penalty[j,j,self.N-1] = self.penalty                
                J += (constraints.T)@self.I_penalty[:,:,self.N-1]@constraints

                return float(J)
        
        def constraint_inter(self,x,u):
                control = (self.u_constraints_matrix@u.reshape(4,1))-self.u_constraints_limits

                #current assumption all obstacle spheres have radius of 0.5m
                obs =  np.zeros((self.n_obs,1))
                quad_position = self.state_to_pos@self.adjust_dim_matrix@x
                for i in range(self.n_obs):
                        obs_dist = quad_position-self.obstacles[i,:]
                        obs[i] = ( (0.5+self.l)**2 ) - (obs_dist.T@obs_dist)

                return np.concatenate((control, obs), axis=0)
        
        def constraint_final(self,x):

                control = -1*np.ones((8,1))
                #current assumption all obstacle spheres have radius of 0.5m
                obs =  np.zeros((self.n_obs,1))
                quad_position = self.state_to_pos@self.adjust_dim_matrix@x
                for i in range(self.n_obs):
                        obs_dist = quad_position-self.obstacles[i,:]
                        obs[i] = ( (0.5+self.l)**2 ) - (obs_dist.T@obs_dist)

                return np.concatenate((control, obs), axis=0)
        
        def constraint_jacobian_x(self,x):

                jac = np.zeros((8+self.n_obs,12))
                quad_position = self.state_to_pos@self.adjust_dim_matrix@x
                for i in range(self.n_obs):
                        interm_var = -2*(self.state_to_pos.T)@(quad_position-self.obstacles[i,:])
                        jac[8+i,:] = interm_var.T

                return jac

        def quat_cost_fn_gradient(self,xtraj,xgoal):
                        
                quat_current = xtraj[3:7]
                quat_goal = xgoal[3:7]
                scalar = -np.sign((quat_goal.T)@quat_current)
                ans = scalar*(quat_goal.T)@self.calc_attitude_jacobian(quat_current)             

                return self.adjust_dim_matrix_quat@ans  

        def quat_cost_fn_hessian(self,xtraj,xgoal):
                        
                quat_current = xtraj[3:7]
                quat_goal = xgoal[3:7]
                scalar = np.sign((quat_goal.T)@quat_current)
                ans = scalar*np.eye(3)*((quat_goal.T)@quat_current)

                return (self.adjust_dim_matrix_quat)@ans@(self.adjust_dim_matrix_quat.T)                    

        def rodrig_to_quat(self,phi):
                phi = np.array(phi).reshape((3,1))
                a = 1/np.sqrt(phi.T@phi)
                return a*np.block([ [1],[phi] ])

        def quat_to_rodrig(self,quat):
                quat = np.array(quat).reshape((4,))
                return quat[1:4]/quat[0]
        
        def backward_pass(self,p,P,d,K,xtraj,utraj):
                # print("Starting Backward Pass")
                del_J = 0.0
                for k in range(self.N-2,-1,-1):
                        # if (k%50==0):
                        #         print(f"{k} out of {self.N-2}")

                        if k in self.way_point_index:
                                Q = self.Q_wp
                                quat_penalty = self.scale_quat_penalty*self.Q_wp[0,0]
                        else:
                                Q = self.Q
                                quat_penalty = self.scale_quat_penalty*self.Q[0,0]

                        #Calculate derivatives
                        q = Q@( self.adjust_dim_matrix@ (xtraj[:,k]-self.xgoal[:,k]) ) 
                        q += quat_penalty*self.quat_cost_fn_gradient(xtraj[:,k],self.xgoal[:,k])

                        r = self.R@(utraj[:,k]-self.uhover)
                
                        A,B = self.calc_jacobians(xtraj[:,k],xtraj[:,k+1],utraj[:,k]) #ForwardDiff.jacobian(dx->dynamics_rk4(dx,utraj[k]),xtraj[:,k])
                

                        constraint_k = self.constraint_inter(xtraj[:,k],utraj[:,k]).squeeze()
                        # constraint_k =  self.constraint_final(xtraj[:,k]).squeeze()
                        constraint_jac_k = self.constraint_jacobian_x(xtraj[:,k])

                        gx = q + A.T@p[:,k+1] + ( constraint_jac_k.T@( self.lg_mul[:,self.N-1] + (self.I_penalty[:,:,k]@constraint_k) ) )
                        gu = r + B.T@p[:,k+1] + ( self.constraint_jac_u.T@( self.lg_mul[:,self.N-1] + (self.I_penalty[:,:,k]@constraint_k) ) )
                        
                        #iLQR (Gauss-Newton) version
                        Gxx = Q + (quat_penalty* self.quat_cost_fn_hessian(xtraj[:,k],self.xgoal[:,k]) )+ (A.T@P[:,:,k+1]@A) + (constraint_jac_k.T@self.I_penalty[:,:,k]@constraint_jac_k)
                        Guu = self.R + (B.T@P[:,:,k+1]@B) + (self.constraint_jac_u.T@self.I_penalty[:,:,k]@self.constraint_jac_u)
                        Gxu = (A.T@P[:,:,k+1]@B) + (constraint_jac_k.T@self.I_penalty[:,:,k]@self.constraint_jac_u)
                        Gux = (B.T@P[:,:,k+1]@A) + (self.constraint_jac_u.T@self.I_penalty[:,:,k]@constraint_jac_k)
                        
                        β = 0.1
                        while True:
                                regul_matrix = np.block([[Gxx,Gxu],[Gux, Guu]])
                                if np.all(np.linalg.eigvals(regul_matrix) > 0):
                                        break
                                elif β>1000000:
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
                
                        del_J += gu.T@d[:,k]
                
                return float(del_J)       

        def calc_trajectory(self):
                print("Starting Trajectory Calculation")
                xic =  jnp.array(self.xic, dtype =float)
                self.uhover = ((1/self.Kf)*self.mass*self.g*0.25)*jnp.ones([4],dtype =float)
                xtraj = np.kron(np.ones((1,self.N)), xic.reshape((13,1)))
                utraj = np.kron(np.ones((1,self.N-1)), self.uhover.reshape((4,1)))
              
                #Initial Rollout
                for i in range(self.N-1):
                        xtraj[:,i+1] = self.dynfn(xtraj[:,i],utraj[:,i])
                        if np.isnan(np.min(xtraj)):
                                print("NANS PRESENT, recheck problem parameters")          

                print("Initial Rollout Done")

                J = self.calc_cost(xtraj,utraj)  

                #iLQR Algorithm
                #Do they really need to be stored for all timesteps?
                p = np.ones((self.nx,self.N))
                P = np.zeros((self.nx,self.nx,self.N))
                d = np.ones((self.nu,self.N-1))
                K = np.zeros((self.nu,self.nx,self.N-1))
                delta_J = 1.0

                xn = np.zeros((self.nx+1,self.N))
                un = np.zeros((self.nu,self.N-1))

                print("Starting AL-iLQR")
                overall_iterations = 0
                cost_prev = J+10000

                constraint_violation = 100

                #As we increase the penalty, the scaling goes whack
                # while (abs(cost_prev-J) > 30) and (overall_iterations < 16) :
                # To avoid the loop breaking during initial cost decrease
                while ( (abs(cost_prev-J) > 10) or (constraint_violation > 10) ) and (overall_iterations < 17):

                        overall_iterations += 1

                        print(f"Starting iLQR loop, overall iteration number {overall_iterations}")
                        print(f"Current cost:{J}, previous cost:{cost_prev}")
                        print(f"Current value of termination criteria 1 of overall algorithim {abs(cost_prev-J)} (required: < 10)")
                
                        cost_prev = J
                        iterations = 0
                        max_prev =1000   

                        p[:,:] = 1
                        P[:,:,:] = 0
                        d[:,:] = 1
                        K[:,:,:] = 0                              

                        xn[:,:] = 0
                        xn[:,0] = xtraj[:,0]
                        un[:,:] = 0             

                        while (np.max(abs(d)) > 0.0005) and ( abs( max_prev-(np.max(abs(d))) )>0.0001):
                                max_prev = np.max(abs(d))

                                xfinal = self.adjust_dim_matrix@(xtraj[:,self.N-1]-self.xgoal[:,self.N-1]) 

                                p[:,self.N-1] = (self.Qf@xfinal) + (self.scale_quat_penalty*self.Qf[0,0]* self.quat_cost_fn_gradient(xtraj[:,self.N-1],self.xgoal[:,self.N-1]) )
                                constraint_N =  self.constraint_final(xtraj[:,self.N-1]).squeeze()
                                constraint_jac_N = self.constraint_jacobian_x(xtraj[:,self.N-1])
                                p[:,self.N-1] += (constraint_jac_N.T)@( self.lg_mul[:,self.N-1] + (self.I_penalty[:,:,self.N-1]@constraint_N) )

                                P[:,:,self.N-1] = self.scale_quat_penalty*self.Qf[0,0]*self.quat_cost_fn_hessian(xtraj[:,self.N-1],self.xgoal[:,self.N-1])
                                P[:,:,self.N-1] += self.Qf  +  ((constraint_jac_N.T)@self.I_penalty[:,:,self.N-1]@constraint_jac_N)                       
                                
                                delta_J = self.backward_pass(p,P,d,K,xtraj,utraj)
                                # print("Backward Pass Done")

                                alpha = 1.0

                                for k in range(self.N-1):
                                        del_x = xn[:,k]-xtraj[:,k]
                                        # print(f"  ")
                                        # print(f"index = {k}")
                                        q_ref = xtraj[3:7,k]
                                        q = xn[3:7,k]
                                        phi = self.quat_to_rodrig(self.L(q_ref).T@q)

                                        del_x = np.concatenate( (del_x[0:3],phi,del_x[7:10],del_x[10:]), axis=0)

                                        un[:,k] = utraj[:,k] - (alpha*d[:,k]) - K[:,:,k]@(del_x)
                                        # print(f"u_k = { un[:,k]}")
                                        # #To prevent NANS and INFS
                                        # un[:,k] = np.clip(un[:,k], 0, self.mass*self.g*self.clip_factor)
                                        
                                        xn[:,k+1] = self.dynfn(xn[:,k],un[:,k])
                                        # print(f"X_k+1 = { xn[:,k+1]}")
                                        if np.isnan(np.min(xn)):
                                                print("NANS PRESENT")                                         
                                # print(f"  ")
                                # print(f"First forward pass done")

                                Jn = self.calc_cost(xn,un) 
                                
                                # while isnan(Jn) || Jn > (J - 1e-2*alpha*ΔJ)
                                while Jn > (J - 1e-2*alpha*delta_J):
                                        alpha = 0.5*alpha
                                        # if alpha == 0.5:
                                        #         print(f"Reducing alpha")
                                        # print(f"Reducing alpha to {alpha}")
                                        for k in range(self.N-1):
                                                del_x = xn[:,k]-xtraj[:,k]
                                                q_ref = xtraj[3:7,k]
                                                q = xn[3:7,k]
                                                phi = self.quat_to_rodrig(self.L(q_ref).T@q)
                                                del_x = np.concatenate( (del_x[0:3],phi,del_x[7:10],del_x[10:]), axis=0)
                                                un[:,k] = utraj[:,k] - (alpha*d[:,k]) - K[:,:,k]@(del_x)

                                                # #To prevent NANS and INFS
                                                # un[:,k] = np.clip(un[:,k], 0, self.mass*self.g*self.clip_factor)

                                                xn[:,k+1] = self.dynfn(xn[:,k],un[:,k])

                                                if np.isnan(np.min(xn)):
                                                        print("NANS PRESENT")  

                                        Jn = self.calc_cost(xn,un) 

                                        if alpha < 0.0000001:
                                                break

                                iterations += 1
                                # print(f"Forward Pass Done, iteration: {iterations}")
                                # print(f"Current tolerance: {np.max(abs(d))}, Required: <0.005 ")
                                # print(f"Other terminataion critera:{abs( max_prev-(np.max(abs(d))) )},Required: <0.0001")
                                J = Jn
                                
                                #Don't use =, will make both the same object
                                # xtraj = np.array(xn)
                                xtraj[:] = xn
                                utraj[:] = un


                        print("iLQR loop finished")
                        self.lg_mul = np.zeros((8+self.n_obs,self.N))
                        constraint_violation = 0

                        for i in range(self.N-1):
                                u = utraj[:,i]
                                constraints = self.constraint_inter(xtraj[:,i],u).squeeze()
                                constraint_violation += sum([x for x in constraints if x > 0])
                                lg_update = self.lg_mul[:,i] + (self.I_penalty[:,:,i]@constraints)
                                self.lg_mul[:,i] = np.maximum(0,lg_update)

                        constraints = self.constraint_final(xtraj[:,self.N-1]).squeeze()
                        constraint_violation += sum([x for x in constraints if x > 0])
                        lg_update = self.lg_mul[:,self.N-1] + (self.I_penalty[:,:,self.N-1]@constraints)
                        self.lg_mul[:,self.N-1] = np.maximum(0,lg_update)
                        self.lg_mul[0:8,self.N-1] = 0
                        self.penalty = self.scale_penalty*self.penalty
                        if self.penalty > 100000:
                                self.penalty = 100000

                return utraj,K,xtraj
