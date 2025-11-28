import yaml
from core.simulator import Simulator
from core.visualizer import Visualizer

# Planners
from planners.dijkstra import Dijkstra_Planner
from planners.a_star import AStar_Planner
from planners.rrt_star import RRTStar_Planner

# Controllers
from controllers.hover_controller import Hover_Controller
from controllers.cascade_pid import Cascade_PID
from controllers.lqr_controller import LQR_Controller
from controllers.mpc_controller import MPC_Controller

with open("config/default.yaml", "r") as f:
    sim_cfg = yaml.safe_load(f)

planner    = Dijkstra_Planner(sim_cfg)
# planner    = AStar_Planner(sim_cfg)
# planner    = RRTStar_Planner(sim_cfg)

controller = Cascade_PID(sim_cfg)
# controller = LQR_Controller(sim_cfg)
# controller = MPC_Controller(sim_cfg)
trajectory, trajectory_object = planner.calculate_trajectory()
controller.set_trajectory(trajectory_object)

# controller = Hover_Controller(sim_cfg)

simulator  = Simulator(sim_cfg)
times, states, controls = simulator.simulate(controller, controller.controller_dt)

# visualizer = Visualizer(sim_cfg, times, states, controls, controller, None)
visualizer = Visualizer(sim_cfg, times, states, controls, controller, planner, trajectory_object)
# #TODO: add option to display tracking error/error data
# visualizer.plot_states()
# #TODO: add legend to animation
visualizer.animate_quadcopter()
