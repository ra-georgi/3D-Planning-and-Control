from controllers import controller

class cascade_PID(controller):

    def __init__(self, cfg):
        super().__init__(cfg)