from copy import copy
import numpy as np

ADRFLC = True

class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        ### TODO implement ESO update
        
        if ADRFLC:
            z = self.state.reshape(6, 1)
            q = q.reshape(2, 1)
            u = u.reshape(2, 1)

            e = q - self.W @ z

            dot_z = self.A @ z + self.B @ u + self.L @ e

            self.state = (z + self.Tp * dot_z).flatten()
            
        ## ADRC
        else:
            z_pred = self.state[0]

            e = q - z_pred

            u_col = np.array([[u]]) if np.isscalar(u) else np.array(u)[:, np.newaxis]
            z_col = self.state[:, np.newaxis] if len(self.state.shape) == 1 else self.state

            f_dot = np.array([[0.0]])

            z_dot = self.A @ z_col + self.B @ u_col + self.W @ f_dot + self.L * (q - z_pred)

            self.state = (z_col + self.Tp * z_dot).flatten()




    def get_state(self):
        return self.state
