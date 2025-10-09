import yaml
from core.simulator import Simulator
from core.visualizer import Visualizer

# Planners
from planners.dijkstra import Dijkstra_Planner

# Controllers
from controllers.hover_controller import Hover_Controller

with open("config/default.yaml", "r") as f:
    sim_cfg = yaml.safe_load(f)

planner    = Dijkstra_Planner(sim_cfg)
controller = Hover_Controller(sim_cfg)
simulator  = Simulator(sim_cfg)

planner.calculate_trajectory()

# times, states, controls = simulator.simulate(controller)

# visualizer = Visualizer(sim_cfg, times, states, controls)
# # visualizer.plot_states()
# visualizer.animate_quadcopter()
