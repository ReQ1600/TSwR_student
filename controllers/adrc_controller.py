import numpy as np
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class ADRController(Controller):
    def __init__(self, Tp, params, adaptive_b=False):
        self.joint_controllers = []
        
        for param in params:
            self.joint_controllers.append(ADRCJointController(*param, Tp))
        
        self.adaptive_b = adaptive_b
        
        self.model = ManiuplatorModel(Tp, m3=0.1, r3=0.05)

    def update_b_from_model(self, x):
        M = self.model.M(x)
        M_inv = np.linalg.inv(M)

        b1 = M_inv[0, 0]
        b2 = M_inv[1, 1]

        self.joint_controllers[0].set_b(b1)
        self.joint_controllers[1].set_b(b2)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):

        u = []


        for i, controller in enumerate(self.joint_controllers):
            joint_state = np.array([
                x[i],
                x[i + 2]
            ])

            u_i = controller.calculate_control(
                joint_state,
                q_d[i],
                q_d_dot[i],
                q_d_ddot[i]
            )

            u.append(u_i)

        if self.adaptive_b:
            self.update_b_from_model(x) 

        return np.array(u)

