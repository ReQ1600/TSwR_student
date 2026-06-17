import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller

FEEDBACK = True

class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)
        self.K_p = 60.0
        self.K_d = 30.0

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """

        q = x[0:2]
        q_dot = x[2:4]

        if FEEDBACK:
            e = q_r - q
            e_dot = q_r_dot - q_dot
            v = q_r_ddot + self.K_d * e_dot + self.K_p * e
        else:
            v = q_r_ddot

        M = self.model.M(x)
        C = self.model.C(x)

        tau = M @ v + C @ q_dot

        return tau.flatten()

