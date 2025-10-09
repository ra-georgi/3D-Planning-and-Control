from planners.planner import Planner
import numpy as np
import yaml

class Dijkstra_Planner(Planner):

    def __init__(self, cfg):
        super().__init__(cfg)
        with open("config/planners/dijkstra.yaml", "r") as f:
            self.planner_params = yaml.safe_load(f)

        self.voxel_resolution = self.planner_params["grid"]["resolution"]
        self.x_limits         = self.planner_params["grid"]["limits"]["x"]
        self.y_limits         = self.planner_params["grid"]["limits"]["y"]
        self.z_limits         = self.planner_params["grid"]["limits"]["z"]
        self.inflation_ratio  = self.planner_params["obstacle_inflation_ratio"]

        self.waypoints =  self.sim_params["world"]["waypoints"]  #List of dictionaries
        self.obstacles =  self.sim_params["obstacles"]["static"]
        self.arm_length = self.sim_params["quadcopter"]["arm_length"]


    def calculate_trajectory(self): #-> str:
        """Calculate and return time parameterized Trajectory"""

        planner_waypoints = []

        # Loop through consecutive pairs of waypoints
        for i in range(len(self.waypoints) - 1):
            start = self.waypoints[i]["pose"][0:3]
            goal  = self.waypoints[i + 1]["pose"][0:3]

            # Generate path between start and goal
            path = self.dijkstra_plan(start, goal)
            planner_waypoints.append(path)
        
        return 
    
    def dijkstra_plan(self, start_pos, goal_pos):

        start_idx = self.points_to_index(start_pos) 
        goal_idx  = self.points_to_index(goal_pos)

        pass


    def points_to_index(self, position):
        """Get grid index for point in space after checking if point is valid"""

        valid = self.check_point_validity(position)
        if not valid:
            return []

        x, y, z = position
        ix = int(round((x - self.x_limits[0]) / self.voxel_resolution))
        iy = int(round((y - self.y_limits[0]) / self.voxel_resolution))
        iz = int(round((z - self.z_limits[0]) / self.voxel_resolution))

        return (ix, iy, iz)


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