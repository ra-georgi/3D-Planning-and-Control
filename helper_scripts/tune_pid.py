import yaml
from core.simulator import Simulator
from core.visualizer import Visualizer
from controllers.cascade_pid2 import Cascade_PID2

with open("config/default.yaml", "r") as f:
    sim_cfg = yaml.safe_load(f)

controller = Cascade_PID2(sim_cfg, True, [0,1,1,0,0,0])


simulator  = Simulator(sim_cfg)
times, states, controls = simulator.simulate(controller, controller.controller_dt)

visualizer = Visualizer(sim_cfg, times, states, controls, controller, None, None)
visualizer.plot_states()
visualizer.animate_quadcopter()
