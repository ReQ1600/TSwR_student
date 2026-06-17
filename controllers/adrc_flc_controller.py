import numpy as np

# from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
# from models.ideal_model import IdealModel
from models.manipulator_model import ManiuplatorModel

class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        self.prev_u = np.zeros(2)
        
        p1 = p[0]
        p2 = p[1]

        self.L = np.array([
            [3.0 * p1,       0.0],
            [0.0,       3.0 * p2],
            [3.0 * p1**2,    0.0],
            [0.0,       3.0 * p2**2],
            [p1**3,          0.0],
            [0.0,            p2**3],
        ])

        W = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ])
    
        A = np.zeros((6, 6))
        B = np.zeros((6, 2))
        
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        q = np.asarray(q, dtype=float).reshape(-1)
        q_dot = np.asarray(q_dot, dtype=float).reshape(-1)

        x = np.concatenate([q, q_dot])

        M = self.model.M(x)
        C = self.model.C(x)

        M_inv = np.linalg.inv(M)

        A = np.zeros((6, 6))

        # q_dot part:
        A[0:2, 2:4] = np.eye(2)

        # q_ddot = -M^-1 C q_dot + M^-1 u + f
        A[2:4, 2:4] = -M_inv @ C
        A[2:4, 4:6] = np.eye(2)

        B = np.zeros((6, 2))
        B[2:4, :] = M_inv

        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        x = np.asarray(x, dtype=float).reshape(-1)

        q = x[:2]

        z_hat_before = self.eso.get_state()
        q_hat_before = z_hat_before[0:2]
        q_dot_hat_before = z_hat_before[2:4]

        self.update_params(q_hat_before, q_dot_hat_before)

        self.eso.update(q, self.prev_u)

        z_hat = self.eso.get_state()

        q_hat = z_hat[0:2]
        q_dot_hat = z_hat[2:4]
        f_hat = z_hat[4:6]

        e = q_d - q_hat
        e_dot = q_d_dot - q_dot_hat

        v = q_d_ddot + self.Kd @ e_dot + self.Kp @ e

        x_hat = np.concatenate([q_hat, q_dot_hat])

        M = self.model.M(x_hat)
        C = self.model.C(x_hat)

        u = M @ (v - f_hat) + C @ q_dot_hat

        self.last_u = u

        return u

