import numpy as np

class Quintic_Spline_Interpolator():
    
    def __init__(self,cfg):
        self.sim_params = cfg
        self.waypoints =  self.sim_params["world"]["waypoints"]  #List of dictionaries

    def interpolate_waypoints(self, planner_waypoints):
        knot_point_times, knot_velocities = self.prepare_knot_points(planner_waypoints)
        self.trajectory = self.compute_splines(planner_waypoints, knot_point_times, knot_velocities)

        return self.trajectory
    
    def prepare_knot_points(self, planner_waypoints):
        """Assign times and velocities to intermediate points (accelerations assumed to be zero)"""
        knot_point_times = []
        knot_velocities  = []
        segment_num = 0

        for segment in planner_waypoints:
            segment_array = np.array(segment)
            distances     = np.linalg.norm(np.diff(segment_array, axis=0), axis=1)
            total_length  = np.sum(distances)
            segment_time_duration = self.waypoints[segment_num+1]["t"]-self.waypoints[segment_num]["t"]
            delta_times   = (segment_time_duration/total_length)*distances
            knot_times    = np.concatenate(([0], np.cumsum(delta_times))) + self.waypoints[segment_num]["t"]  

            knot_point_times.append(knot_times)

            # Assign velocities based on central difference derivative
            inter_velocities = []
            for point_num, point in enumerate(segment):

                if point_num == 0:
                    inter_velocities.append(np.array(self.waypoints[segment_num]["pose"][7:10]))
                    continue
                elif point == segment[-1]:
                    inter_velocities.append(np.array(self.waypoints[segment_num+1]["pose"][7:10]))
                    continue
                
                point_previous = np.array(segment[point_num-1])
                point_next     = np.array(segment[point_num+1])
                time_previous  = knot_times[point_num-1]
                time_next      = knot_times[point_num+1]


                velocity = (point_next - point_previous)/(time_next - time_previous)
                inter_velocities.append(velocity)


            knot_velocities.append(inter_velocities)
            segment_num += 1

        return knot_point_times, knot_velocities

    def compute_splines(self, planner_waypoints, knot_point_times, knot_velocities):
        
        for segment_num, segment in enumerate(planner_waypoints):
            for wp_num, waypoint in enumerate(segment):

                if wp_num == (len(segment)-1):
                    continue

                x_initial, y_initial, z_initial = waypoint
                x_final,   y_final,   z_final   = segment[wp_num+1]
                t_initial = knot_point_times[segment_num][wp_num]
                t_final   = knot_point_times[segment_num][wp_num+1]
                vx_initial, vy_initial, vz_initial = knot_velocities[segment_num][wp_num]
                vx_final,   vy_final,   vz_final   = knot_velocities[segment_num][wp_num+1]

                # Considering acclerations to be zero at boundaries   #TODO: need to calculate this
                acc_x_init  = acc_y_init  = acc_z_init  = 0
                acc_x_final = acc_y_final = acc_z_final = 0

                self.compute_spline_coefficients(x_initial,x_final,vx_initial,vx_final,acc_x_init,acc_x_final,t_initial,t_final)
                self.compute_spline_coefficients(y_initial,y_final,vy_initial,vy_final,acc_y_init,acc_y_final,t_initial,t_final)
                self.compute_spline_coefficients(z_initial,z_final,vz_initial,vz_final,acc_z_init,acc_z_final,t_initial,t_final)


    def compute_spline_coefficients(self, p_i, p_f, v_i, v_f, a_i, a_f, t_i, t_f):
        pass