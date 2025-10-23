from planners.planner import Planner
import numpy as np
import yaml
from collections import defaultdict
import heapq
from planners.interpolators.quintic_spline import Quintic_Spline_Interpolator

class Node:
    # __slots__ = ("pos", "parent")
    def __init__(self, position, parent=None):
        self.position =  position
        self.parent   =  parent


class RRTStar_Planner(Planner):

    def __init__(self, cfg):
        super().__init__(cfg)
        with open("config/planners/rrt_star.yaml", "r") as f:
            self.planner_params = yaml.safe_load(f)

        self.voxel_resolution = self.planner_params["grid"]["resolution"]
        self.x_limits         = self.planner_params["grid"]["limits"]["x"]
        self.y_limits         = self.planner_params["grid"]["limits"]["y"]
        self.z_limits         = self.planner_params["grid"]["limits"]["z"]
        self.inflation_ratio  = self.planner_params["obstacle_inflation_ratio"]

        # RRT hyper-parameters
        self.propagation_distance_m   = self.planner_params["propagation_distance_m"]
        self.max_itertions            = self.planner_params["max_itertions"]
        self.goal_sample_rate         = self.planner_params["goal_sample_rate"]
        self.goal_threshold_radius    = self.planner_params["goal_threshold_radius"]
        self.collision_check_distance = self.planner_params["collision_check_distance"]

        self.waypoints =  self.sim_params["world"]["waypoints"]  #List of dictionaries
        self.obstacles =  self.sim_params["obstacles"]["static"]
        self.arm_length = self.sim_params["quadcopter"]["arm_length"]

        self.neighbor_deltas = self.generate_delta_values()
        self.trajectory_generator = Quintic_Spline_Interpolator(cfg)
        
        self.planner_name = "RRT-Star"
        self.interpolator_name = "Quintic Spline"

    def calculate_trajectory(self): #-> str:
        """Calculate and return time parameterized Trajectory"""

        planner_waypoints = []
        print("Starting RRT* for path planning")

        # Loop through consecutive pairs of waypoints
        for i in range(len(self.waypoints) - 1):
            start = self.waypoints[i]["pose"][0:3]
            goal  = self.waypoints[i + 1]["pose"][0:3]

            # Generate path between start and goal
            path = self.rrt_star_plan(start, goal)
            planner_waypoints.append(path)
        
        print("Converting RRT* waypoints to trajectory")
        trajectory = self.trajectory_generator.interpolate_waypoints(planner_waypoints)

        return trajectory, self.trajectory_generator #To evaluate trajectories, feed to controller
    
    def rrt_star_plan(self, start_pos, goal_pos):

        if (not self.check_point_validity(start_pos)) or (not self.check_point_validity(goal_pos)):
            print("Start or goal is invalid/outside bounds or in collision.")
            return []

        nodes = [Node(start_pos, parent=None)]
        goal_pos = np.asarray(goal_pos, dtype=float)

        for iteration in range(self.max_iters):
            sampled_position = self.sample_space(goal_pos)

        print("No path found with RRT*")
        return []


    def check_point_validity(self, position):

        position =  np.array(position)

        # Check if point is within planner bounds
        bounds = np.array([
            self.x_limits,   # x bounds
            self.y_limits,   # y bounds
            self.z_limits    # z bounds
        ])
        inside = np.all((position >= bounds[:,0]) & (position <= bounds[:,1]))   

        if (not inside):
            print("Waypoint outside bounds, aborting ...")
            return False

        # Check if point does not lead to collision
        for obstacle in self.obstacles:
            radius_obstacle   = obstacle["radius"]
            position_obstacle = obstacle["pose"]

            x, y, z = position
            cx, cy, cz = position_obstacle

            distance = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
            # Collision sphere radius is increased by inflation_ratio * length of quadcopter arm
            inflated_radius = radius_obstacle + (self.inflation_ratio*self.arm_length)

            if distance <= inflated_radius*inflated_radius:
                return False

        return True

    def sample_space(self, goal_pos):
        # if np.random.rand() < self.goal_sample_rate:
        #     return goal_pos.copy()
        # # uniform in bounds
        # x = np.random.uniform(self.x_limits[0], self.x_limits[1])
        # y = np.random.uniform(self.y_limits[0], self.y_limits[1])
        # z = np.random.uniform(self.z_limits[0], self.z_limits[1])
        # return np.array([x, y, z], dtype=float)        


    # def construct_astar_path(self, parent_dict, current_idx):
    #     path_idx_list = [current_idx]
    #     while current_idx in parent_dict:
    #         current_idx = parent_dict[current_idx]
    #         path_idx_list.append(current_idx)
    #     path_idx_list.reverse()

    #     path_points = []
    #     for i in path_idx_list:
    #         path_points.append(self.index_to_point(i))

    #     return path_points
    
    # @staticmethod
    # def generate_delta_values():
    #     # 26-connected
    #     deltas = []
    #     for dx in (-1, 0, 1):
    #         for dy in (-1, 0, 1):
    #             for dz in (-1, 0, 1):
    #                 if dx == dy == dz == 0:
    #                     continue
    #                 deltas.append((dx, dy, dz))
    #     return deltas        

    # def path_cost(self, current_idx, neighbor_idx):
    #     """ Returns cost as Euclidean distance between voxels in meters"""

    #     dx = (neighbor_idx[0] - current_idx[0]) * self.voxel_resolution
    #     dy = (neighbor_idx[1] - current_idx[1]) * self.voxel_resolution
    #     dz = (neighbor_idx[2] - current_idx[2]) * self.voxel_resolution

    #     return np.sqrt(dx*dx + dy*dy + dz*dz)

    # # def heuristic_cost(self, current_idx, goal_idx):
    #     """ Same as path cost but separate function just to differentiate use"""

    #     dx = (goal_idx[0] - current_idx[0]) * self.voxel_resolution
    #     dy = (goal_idx[1] - current_idx[1]) * self.voxel_resolution
    #     dz = (goal_idx[2] - current_idx[2]) * self.voxel_resolution

    #     return np.sqrt(dx*dx + dy*dy + dz*dz)
    


    # def point_to_index(self, position):
    #     """Get grid index for point in space after checking if point is valid"""

    #     valid = self.check_point_validity(position)
    #     if not valid:
    #         return []

    #     x, y, z = position
    #     ix = int(round((x - self.x_limits[0]) / self.voxel_resolution))
    #     iy = int(round((y - self.y_limits[0]) / self.voxel_resolution))
    #     iz = int(round((z - self.z_limits[0]) / self.voxel_resolution))

    #     return (ix, iy, iz)
    
    # def index_to_point(self, index):
    #     ix, iy, iz = index
    #     x = self.x_limits[0] + (ix * self.voxel_resolution)
    #     y = self.y_limits[0] + (iy * self.voxel_resolution)
    #     z = self.z_limits[0] + (iz * self.voxel_resolution)
    #     return (x, y, z)        
