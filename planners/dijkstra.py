from planners.planner import Planner
import yaml

class Dijkstra(Planner):

    def __init__(self, cfg):
        super().__init__(cfg)
        with open("config/planners/dijkstra.yaml", "r") as f:
            self.planner_params = yaml.safe_load(f)

        self.voxel_resolution = self.planner_params["grid"]["resolution"]
        self.x_limits         = self.planner_params["grid"]["limits"]["x"]
        self.y_limits         = self.planner_params["grid"]["limits"]["y"]
        self.z_limits         = self.planner_params["grid"]["limits"]["z"]

        self.waypoints = self.sim_params["world"]["waypoints"]  #List of dictionaries

    def calculate_trajectory(self): #-> str:
        """Calculate and return time parameterized Trajectory"""

        planner_waypoints = []

        # Loop through consecutive pairs of waypoints
        for i in range(len(self.waypoints) - 1):
            start = self.waypoints[i]["pose"][0:3]
            goal  = self.waypoints[i + 1]["pose"][0:3]

            print(start)
            print(goal)

            # # Generate path between start and goal
            # path = self.dijkstra_planner(start, goal)
            # planner_waypoints.append(path)
        
        return 
    
    def dijkstra_planner(self):
        pass
