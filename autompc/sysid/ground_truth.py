import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from pdb import set_trace
from sklearn.linear_model import  Lasso

from .model import Model
import mujoco_py

class GroundTruth(Model):
    def __init__(self, system, env):
        super().__init__(system)
        self.env = env
        self.fin_step = 1e-6

    @property
    def state_dim(self):
        return self.system.obs_dim

    def train(self, trajs):
        pass

    def traj_to_state(self, traj):
        state = np.zeros((self.system.obs_dim,))
        state[:] = traj[-1].obs[:]
        return state[:]

    def update_state(state, new_obs, new_ctrl):
        return state[:]

    def pred(self, x, u, n_frames=5):
        old_state = self.env.sim.get_state()
        old_qpos = old_state[1]
        qpos = x[:len(old_qpos)]
        qvel = x[len(old_qpos):]
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                old_state.act, old_state.udd_state)
        self.env.sim.set_state(new_state)
        #env.sim.forward()
        self.env.sim.data.ctrl[:] = u
        for _ in range(n_frames):
            self.env.sim.step()
        new_qpos = self.env.sim.data.qpos
        new_qvel = self.env.sim.data.qvel

        return np.concatenate([new_qpos, new_qvel])

    def pred_diff(self, state, ctrl):
        m = ctrl.shape[0]
        n = state.shape[0]
        Jac_state = np.zeros((n,n))
        Jac_ctrl = np.zeros((n,m))
        for si in range(n):            
            for sj in range(n):
                ej = np.zeros(n)
                ej[sj] = 1
                f1 = self.pred(state + self.fin_step * ej, ctrl)
                f2 = self.pred(state - self.fin_step * ej, ctrl)
                Jac_state[si, sj] = (f1 - f2)[si] / (2 * self.fin_step)

            for cj in range(m):
                ej = np.zeros(m)
                ej[cj] = 1
                f1 = self.pred(state, ctrl + self.fin_step * ej)
                f2 = self.pred(state, ctrl - self.fin_step * ej)
                Jac_ctrl[si, cj] = (f1 - f2)[si] / (2 * self.fin_step)

        new_state = self.pred(state, ctrl)

        return new_state, Jac_state, Jac_ctrl

    @staticmethod
    def get_configuration_space(system):
        """
        Returns the model configuration space.
        """
        return None
