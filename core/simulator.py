import numpy as np

class Simulator():

    def __init__(self, cfg):
        self.params = cfg

    def simulate(self,controller): #-> str:
        """Simulate Quadcopter Flight"""

        dt = self.params["time"]["dt"]
        tf = self.params["time"]["duration"]
        waypoints =  self.params["world"]["waypoints"]

        n_steps = int(tf/dt)
        x0 = np.array(waypoints[0]["pose"], dtype=float)

        states   = np.zeros([13, n_steps])
        times    = np.zeros([n_steps])
        controls = np.zeros([4, n_steps])

        states[:, 0] = x0

        for k in range(n_steps):
            t = k * dt
            u = controller(x, t)                # compute control
            dx = system(x, u, t)                # system dynamics
            x = x + dx * dt                     # simple Euler integration

            times.append(t + dt)
            states.append(x.copy())
            controls.append(u)

        # return np.array(times), np.array(states), np.array(controls)
        
    def take_step(self):
        """ Simulate for a single time step"""

    def rk4_step(self):
        """" Lol"""





