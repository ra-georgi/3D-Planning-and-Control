from planners.planner import Planner
import numpy as np
import yaml
from collections import defaultdict
import heapq
from planners.interpolators.quintic_spline import Quintic_Spline_Interpolator

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

        start_idx = self.point_to_index(start_pos) 
        goal_idx  = self.point_to_index(goal_pos)

        cost_dict   = defaultdict(lambda: float("inf"))
        cost_dict[start_idx] = 0.0          
        parent_dict = {}

        heap_list = []
        tiebreaker_index = 0      # For creating deterministic paths
        start_heuristic_cost = self.heuristic_cost(start_idx, goal_idx)
        # Use heuristic cost + path cost in heap to explore the most promising (least expensive) nodes
        heapq.heappush(heap_list, (0.0+start_heuristic_cost, tiebreaker_index, start_idx))
        visited_indices = set()

        while heap_list:
            current_cost, _, current_idx = heapq.heappop(heap_list)

            if current_idx in visited_indices:
                continue
            visited_indices.add(current_idx)

            if current_idx == goal_idx:
                return self.construct_astar_path(parent_dict, current_idx)
            
            for dx in self.neighbor_deltas:

                neighbor_idx = (current_idx[0]+dx[0], 
                                current_idx[1]+dx[1], 
                                current_idx[2]+dx[2])
                
                if not self.check_point_validity(self.index_to_point(neighbor_idx)):
                    continue

                path_cost = self.path_cost(current_idx, neighbor_idx)  # Euclidean in grid (scaled by res)
                potential_cost = current_cost + path_cost
                if potential_cost < cost_dict[neighbor_idx]:
                    cost_dict[neighbor_idx]   = potential_cost
                    parent_dict[neighbor_idx] = current_idx
                    tiebreaker_index += 1
                    heuristic_cost = self.heuristic_cost(neighbor_idx, goal_idx)
                    heapq.heappush(heap_list, (potential_cost + heuristic_cost, tiebreaker_index, neighbor_idx))
                    
        print("No path found with RRT*")
        return []


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

    # def check_point_validity(self, position):

    #     position =  np.array(position)

    #     # Check if point is within planner bounds
    #     bounds = np.array([
    #         self.x_limits,   # x bounds
    #         self.y_limits,   # y bounds
    #         self.z_limits    # z bounds
    #     ])
    #     inside = np.all((position >= bounds[:,0]) & (position <= bounds[:,1]))   

    #     if (not inside):
    #         print("Waypoint outside bounds, aborting ...")
    #         return False

    #     # Check if point does not lead to collision
    #     for obstacle in self.obstacles:
    #         radius_obstacle   = obstacle["radius"]
    #         position_obstacle = obstacle["pose"]

    #         x, y, z = position
    #         cx, cy, cz = position_obstacle

    #         distance = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
    #         # Collision sphere radius is increased by inflation_ratio * length of quadcopter arm
    #         inflated_radius = radius_obstacle + (self.inflation_ratio*self.arm_length)

    #         if distance <= inflated_radius*inflated_radius:
    #             return False

    #     return True
    
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

    # def heuristic_cost(self, current_idx, goal_idx):
        """ Same as path cost but separate function just to differentiate use"""

        dx = (goal_idx[0] - current_idx[0]) * self.voxel_resolution
        dy = (goal_idx[1] - current_idx[1]) * self.voxel_resolution
        dz = (goal_idx[2] - current_idx[2]) * self.voxel_resolution

        return np.sqrt(dx*dx + dy*dy + dz*dz)