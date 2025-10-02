import yaml
from core.simulator import Simulator

# Controllers
from controllers.hover_controller import Hover_Controller

with open("config/default.yaml", "r") as f:
    sim_cfg = yaml.safe_load(f)

controller = Hover_Controller(sim_cfg)
simulator  = Simulator(sim_cfg)

times, states, controls = simulator.simulate(controller)
print(times)
print()
print(states)
print()
print(controls)
print()

