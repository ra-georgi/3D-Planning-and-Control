import yaml
from core.simulator import Simulator
from core.visualizer import Visualizer

# Planners
from planners.dijkstra import Dijkstra_Planner

# Controllers
from controllers.hover_controller import Hover_Controller
from controllers.cascade_pid import Cascade_PID

with open("config/default.yaml", "r") as f:
    sim_cfg = yaml.safe_load(f)

planner    = Dijkstra_Planner(sim_cfg)
controller = Cascade_PID(sim_cfg)
trajectory, trajectory_object = planner.calculate_trajectory()
controller.set_trajectory(trajectory_object)

# controller = Hover_Controller(sim_cfg)

simulator  = Simulator(sim_cfg)
times, states, controls = simulator.simulate(controller, controller.controller_dt)

visualizer = Visualizer(sim_cfg, times, states, controls)
# #TODO: add option to display tracking error/error data
visualizer.plot_states()
# #TODO: add legend to animation
visualizer.animate_quadcopter()
