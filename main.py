import yaml
from core import Simulator
# Controllers
from controllers import Hover_Controller

with open("config/default.yaml", "r") as f:
    sim_cfg = yaml.safe_load(f)

controller = Hover_Controller(sim_cfg)
simulator = Simulator(sim_cfg)

simulator.simulate(controller)



