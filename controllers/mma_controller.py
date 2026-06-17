import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.Tp = Tp
        self.models = [
            ManiuplatorModel(Tp, 0.1),
            ManiuplatorModel(Tp, 0.01, 0.01),
            ManiuplatorModel(Tp, 1.0, 0.3)
        ]
        
        self.i = 0
        
        self.prev_x = None
        self.prev_u = np.zeros(2)

        self.Kd = 30
        self.Kp = 60


    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        if self.prev_x is None:
            self.prev_x = x.copy()
            return
        
        q_prev = self.prev_x[:2]
        q_dot_prev = self.prev_x[2:]

        e = []

        for model in self.models:
            M = model.M(self.prev_x)
            C = model.C(self.prev_x)

            q_ddot_pred = np.linalg.solve(M, self.prev_u - C @ q_dot_prev)
            q_dot_pred = q_dot_prev + self.Tp * q_ddot_pred
            q_pred = q_prev + self.Tp * q_dot_prev + 0.5 * self.Tp ** 2 * q_ddot_pred        

            x_pred = np.concatenate([q_pred, q_dot_pred])

            e.append(np.linalg.norm(x-x_pred))

        self.i = int(np.argmin(e))
        self.prev_x = x.copy()

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        e = q_r - q
        e_dot = q_r_dot - q_dot
        v = q_r_ddot + self.Kd * e_dot + self.Kp * e
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        u = u.flatten()
        self.prev_u = u.copy()
        return u
