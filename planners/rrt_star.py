from planners.planner import Planner
import numpy as np
import yaml
from planners.interpolators.quintic_spline import Quintic_Spline_Interpolator

class Node:
    # __slots__ = ("position", "parent", "cost")
    def __init__(self, position, parent=None, cost=np.inf):
        self.position =  position # parent index in nodes
        self.parent   =  parent 
        self.cost     =  cost


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
        self.max_iterations            = self.planner_params["max_iterations"]
        self.goal_sample_rate         = self.planner_params["goal_sample_rate"]
        self.goal_threshold_radius    = self.planner_params["goal_threshold_radius"]
        self.collision_check_distance = self.planner_params["collision_segment_distance"]

        self.waypoints =  self.sim_params["world"]["waypoints"]  #List of dictionaries
        self.obstacles =  self.sim_params["obstacles"]["static"]
        self.arm_length = self.sim_params["quadcopter"]["arm_length"]

        self.trajectory_generator = Quintic_Spline_Interpolator(cfg)
        
        self.planner_name = "RRT*"
        self.interpolator_name = "Quintic Spline"

        self.gamma = self.calculate_gamma()

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
        
        start_pos = np.asarray(start_pos, dtype=float)
        goal_pos  = np.asarray(goal_pos, dtype=float)
        nodes = [Node(start_pos, None, 0.0)]

        best_goal_idx = None
        best_goal_cost = np.inf

        for iteration in range(self.max_iterations):
            sampled_position      = self.sample_space(goal_pos)
            idx_nearest           = self.find_nearest_node(nodes, sampled_position)
            nearest_node_position = nodes[idx_nearest].position
            new_position          = self.get_new_node_position(nearest_node_position, sampled_position)
            
            if (new_position is None) or (not self.check_segment_free(nearest_node_position, new_position)):
                continue

            # Choose best parent among neighbors
            neighbor_indices = self.get_neighbors(nodes, new_position)
            new_node_cost    = nodes[idx_nearest].cost + np.linalg.norm(new_position - nodes[idx_nearest].position)
            new_node_parent  = idx_nearest

            for j in neighbor_indices:
                if self.check_segment_free(nodes[j].position, new_position):
                    candidate_cost = nodes[j].cost + np.linalg.norm(new_position - nodes[j].position)
                    if candidate_cost < new_node_cost:
                        new_node_cost = candidate_cost
                        new_node_parent = j

            nodes.append(Node(new_position, new_node_parent, new_node_cost))
            new_node_idx = len(nodes) - 1

            # Rewire neighbors through the new node if beneficial
            for j in neighbor_indices:
                if j == new_node_parent:
                    continue
                if self.check_segment_free(nodes[new_node_idx].position, nodes[j].position):
                    new_cost = nodes[new_node_idx].cost + np.linalg.norm(nodes[j].position - nodes[new_node_idx].position)
                    if new_cost < nodes[j].cost:
                        nodes[j].parent = new_node_idx
                        nodes[j].cost   = new_cost


            # Check goal reached
            if np.linalg.norm(new_position - goal_pos) <= self.goal_threshold_radius:
                # Try to connect straight to goal
                if self.check_segment_free(new_position, goal_pos):
                    goal_cost = nodes[new_node_idx].cost + np.linalg.norm(goal_pos - nodes[new_node_idx].position)
                    if goal_cost < best_goal_cost:
                        nodes.append(Node(goal_pos, new_node_idx, goal_cost))
                        best_goal_idx = len(nodes) - 1
                        best_goal_cost = goal_cost
                        # For early exit
                        # return self.construct_rrt_star_path(nodes, len(nodes)-1)


        if best_goal_idx is not None:
            return self.construct_rrt_star_path(nodes, best_goal_idx)

        # If exact goal not reached, pick best near-goal node and try to connect once
        # TODO: But what about the case where this node is very far from goal? How would it affect interpolation for trajectory
        best_idx = np.argmin([np.linalg.norm(n.position - goal_pos) for n in nodes])
        if self.check_segment_free(nodes[best_idx].position, goal_pos):
            nodes.append(Node(goal_pos, parent=best_idx, 
                              cost=nodes[best_idx].cost + np.linalg.norm(goal_pos - nodes[best_idx].position)
                              ))
            return self.construct_rrt_star_path(nodes, len(nodes)-1)
              
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
        if np.random.rand() < self.goal_sample_rate:
            return goal_pos.copy()
        # uniform in bounds
        x = np.random.uniform(self.x_limits[0], self.x_limits[1])
        y = np.random.uniform(self.y_limits[0], self.y_limits[1])
        z = np.random.uniform(self.z_limits[0], self.z_limits[1])
        return np.array([x, y, z], dtype=float)        


    def find_nearest_node(self, nodes, sampled_position):
        distances = [np.linalg.norm(node.position-sampled_position) for node in nodes ]
        return np.argmin(distances)

    def get_new_node_position(self, nearest_node_position, sampled_position):
        direction = sampled_position - nearest_node_position
        distance  = np.linalg.norm(direction)

        if distance == 0.0: 
            return None

        unit_vector = direction/np.linalg.norm(direction)

        if distance > self.propagation_distance_m:
            step_size = self.propagation_distance_m
        else:
            step_size = distance

        return nearest_node_position + (step_size*unit_vector)

    def check_segment_free(self, nearest_node_position, new_position):

        if (not self.check_point_validity(nearest_node_position)) or (not self.check_point_validity(new_position)):
            return False
        
        direction = new_position - nearest_node_position
        distance = np.linalg.norm(direction)

        n_steps = max(2,int(np.ceil(distance / self.collision_check_distance)))

        for i in range(n_steps + 1):
            alpha = i / n_steps
            p = nearest_node_position + (alpha * direction)
            if not self.check_point_validity(p):
                return False
        return True        

    def construct_rrt_star_path(self, nodes, goal_idx):
        path = []
        current_idx = goal_idx
        while current_idx is not None:
            path.append(tuple(nodes[current_idx].position))
            current_idx = nodes[current_idx].parent
        path.reverse()

        return path
    
    def get_neighbors(self, nodes, new_position):
    # Implementing as per https://in.mathworks.com/help/nav/ug/calculating-appropriate-ball-radius-constant-for-plannerrrtstar.html
        N = max(1, len(nodes))
        d = 3  # 3D

        radius_ball = min(self.propagation_distance_m, 
                        (self.gamma * (np.log(N + 1) / (N + 1))) ** (1.0 / d)  
                        )
        
        idx = []
        for i, node in enumerate(nodes):
            if np.linalg.norm(node.position - new_position) <= radius_ball:
                idx.append(i)
        return idx

    def calculate_gamma(self):
        
        V_unit_ball = (4/3)*(np.pi)
        Lx = self.x_limits[1]-self.x_limits[0]
        Ly = self.y_limits[1]-self.y_limits[0]
        Lz = self.z_limits[1]-self.z_limits[0]

        total_volume    = Lx * Ly * Lz
        obstacle_volume = 0

        for obstacle in self.obstacles:
            radius_obstacle   = obstacle["radius"]
            # Collision sphere radius is increased by inflation_ratio * length of quadcopter arm
            inflated_radius =  radius_obstacle + (self.inflation_ratio*self.arm_length)
            obstacle_volume += (4/3)*(np.pi)*(inflated_radius**3)

        V_free = total_volume - obstacle_volume
        dim = 3
        return (2.0 ** dim) * (1.0 + (1.0 / dim)) * (V_free / V_unit_ball)