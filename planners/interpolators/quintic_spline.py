import numpy as np

class Quintic_Spline_Interpolator():
    
    #TODO: Recheck accelerations for intermediate planner waypoints, allow non zero accleration at these points
    def __init__(self,cfg):
        self.sim_params = cfg
        self.waypoints  = self.sim_params["world"]["waypoints"]  #List of dictionaries

    def interpolate_waypoints(self, planner_waypoints):
        knot_point_times, knot_velocities, knot_accelerations = self.prepare_knot_points(planner_waypoints)
        self.trajectory = self.compute_splines(planner_waypoints, knot_point_times, knot_velocities, knot_accelerations)

        return self.trajectory
    
    def prepare_knot_points(self, planner_waypoints):
        """Assign times and velocities to intermediate points (accelerations assumed to be zero)"""
        knot_point_times    = []
        knot_velocities     = []
        knot_accelerations  = []
        segment_num = 0

        for segment in planner_waypoints:
            segment_array = np.array(segment)
            distances     = np.linalg.norm(np.diff(segment_array, axis=0), axis=1)
            total_length  = np.sum(distances)
            segment_time_duration = self.waypoints[segment_num+1]["t"]-self.waypoints[segment_num]["t"]
            delta_times   = (segment_time_duration/total_length)*distances
            knot_times    = np.concatenate(([0], np.cumsum(delta_times))) + self.waypoints[segment_num]["t"]  

            knot_point_times.append(knot_times)

            # Assign velocities & accelerations based on central difference derivative
            inter_velocities    = []
            inter_accelerations = []

            for point_num, point in enumerate(segment):
                if point_num == 0:
                    inter_velocities.append(np.array(self.waypoints[segment_num]["pose"][7:10]))
                    inter_accelerations.append(np.array([0, 0, 0]))
                    continue
                elif point == segment[-1]:
                    inter_velocities.append(np.array(self.waypoints[segment_num+1]["pose"][7:10]))
                    inter_accelerations.append(np.array([0, 0, 0]))
                    continue
                
                point_previous = np.array(segment[point_num-1])
                point_next     = np.array(segment[point_num+1])
                time_previous  = knot_times[point_num-1]
                time_next      = knot_times[point_num+1]

                velocity = (point_next - point_previous)/(time_next - time_previous)

                time_current = knot_times[point_num]
                inter_velocities.append(velocity)

                v_fwd = (point_next - point)/(time_next - time_current)
                v_bck = (point - point_previous)/(time_current - time_previous)

                accel = (2*(v_fwd-v_bck)) / (time_next-time_previous)
                inter_accelerations.append(accel)


            knot_velocities.append(inter_velocities)
            knot_accelerations.append(inter_accelerations)
            segment_num += 1

        return knot_point_times, knot_velocities, knot_accelerations

    def compute_splines(self, planner_waypoints, knot_point_times, knot_velocities, knot_accelerations):
        
        trajectory = []
        for segment_num, segment in enumerate(planner_waypoints):
            for wp_num, waypoint in enumerate(segment):
                trajectory_segment = {}

                if wp_num == (len(segment)-1):
                    continue

                x_initial, y_initial, z_initial = waypoint
                x_final,   y_final,   z_final   = segment[wp_num+1]

                t_initial = knot_point_times[segment_num][wp_num]
                t_final   = knot_point_times[segment_num][wp_num+1]

                vx_initial, vy_initial, vz_initial      = knot_velocities[segment_num][wp_num]
                vx_final,   vy_final,   vz_final        = knot_velocities[segment_num][wp_num+1]
                acc_x_init, acc_y_init, acc_z_init      = knot_accelerations[segment_num][wp_num]
                acc_x_final, acc_y_final, acc_z_final   = knot_accelerations[segment_num][wp_num+1]

                trajectory_segment["t0"]    = t_initial
                trajectory_segment["tf"]    = t_final
                trajectory_segment["coeff"] = np.zeros((3,6))

                trajectory_segment["coeff"][0,:] = self.compute_spline_coefficients(x_initial,x_final,vx_initial,vx_final,acc_x_init,acc_x_final,t_initial,t_final)
                trajectory_segment["coeff"][1,:] = self.compute_spline_coefficients(y_initial,y_final,vy_initial,vy_final,acc_y_init,acc_y_final,t_initial,t_final)
                trajectory_segment["coeff"][2,:] = self.compute_spline_coefficients(z_initial,z_final,vz_initial,vz_final,acc_z_init,acc_z_final,t_initial,t_final)

                trajectory.append(trajectory_segment)

        return trajectory


    def compute_spline_coefficients(self, p_i, p_f, v_i, v_f, a_i, a_f, t_i, t_f):

        T  = t_f - t_i
        d0 = p_f - (p_i+(v_i*T)+(0.5*a_i*T*T))
        d1 = v_f - (v_i + (a_i*T)) 
        d2 = a_f - a_i

        c0 = p_i
        c1 = v_i
        c2 = 0.5*a_i
        c3 = ((10*d0)/(T**3)) - ((4*d1)/(T**2)) + ((0.5*d2)/T)
        c4 = ((-15*d0)/(T**4)) + ((7*d1)/(T**3)) - (d2/(T**2))
        c5 = ((6*d0)/(T**5)) - ((3*d1)/(T**4)) + ((0.5*d2)/(T**3))

        return np.array([c0, c1, c2, c3, c4, c5])
    
    def evaluate_trajectory(self, time):

        tau = -1
        # Find which quintic polynomial corresponds to this time
        for segment in self.trajectory:
            if segment['t0'] <=  time < segment['tf']:
                coeff = segment['coeff']
                tau   = time - segment['t0']
        # May not be necessary
        if time == self.trajectory[-1]['tf']:
            coeff = self.trajectory[-1]['coeff']
            tau   = time - self.trajectory[-1]['t0']

        if tau == -1:
            # if t is outside the defined time range
            print( f"Time outside trajectory time limits: {time}")
            return None, None, None
        else:
            coeff_x = coeff[0,:]
            coeff_y = coeff[1,:]
            coeff_z = coeff[2,:]

            pos = np.array([self.eval_poly(coeff_x, tau),
                            self.eval_poly(coeff_y, tau),
                            self.eval_poly(coeff_z, tau)])
            vel = np.array([self.eval_der_poly(coeff_x, tau),
                            self.eval_der_poly(coeff_y, tau),
                            self.eval_der_poly(coeff_z, tau)])
            acc = np.array([self.eval_2nd_der_poly(coeff_x, tau),
                            self.eval_2nd_der_poly(coeff_y, tau),
                            self.eval_2nd_der_poly(coeff_z, tau)])
            
            return pos, vel, acc


    def evaluate_jerk_snap(self, time):

        tau = -1
        # Find which quintic polynomial corresponds to this time
        for segment in self.trajectory:
            if segment['t0'] <=  time < segment['tf']:
                coeff = segment['coeff']
                tau   = time - segment['t0']
        # May not be necessary
        if time == self.trajectory[-1]['tf']:
            coeff = self.trajectory[-1]['coeff']
            tau   = time - self.trajectory[-1]['t0']

        if tau == -1:
            # if t is outside the defined time range
            print("Time outside trajectory time limits")
            return None, None, None
        else:
            coeff_x = coeff[0,:]
            coeff_y = coeff[1,:]
            coeff_z = coeff[2,:]

            jerk = np.array([self.eval_3rd_der_poly(coeff_x, tau),
                             self.eval_3rd_der_poly(coeff_y, tau),
                             self.eval_3rd_der_poly(coeff_z, tau)])
            snap = np.array([self.eval_4th_der_poly(coeff_x, tau),
                             self.eval_4th_der_poly(coeff_y, tau),
                             self.eval_4th_der_poly(coeff_z, tau)])       

            return jerk, snap

    @staticmethod
    def eval_poly(c, tau):
        return (c[0] + c[1]*tau + c[2]*tau**2 + c[3]*tau**3 + c[4]*tau**4 + c[5]*tau**5)

    @staticmethod
    def eval_der_poly(c, tau):
        return (c[1] + 2*c[2]*tau + 3*c[3]*tau**2 + 4*c[4]*tau**3 + 5*c[5]*tau**4)

    @staticmethod
    def eval_2nd_der_poly(c, tau):
        return (2*c[2] + 6*c[3]*tau + 12*c[4]*tau**2 + 20*c[5]*tau**3)
    
    @staticmethod
    def eval_3rd_der_poly(c, tau):
        return (6*c[3] + 24*c[4]*tau + 60*c[5]*tau**2)
    
    @staticmethod
    def eval_4th_der_poly(c, tau):
        return (24*c[4] + 120*c[5]*tau)