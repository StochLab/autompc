# Created by William Edwards (wre2@illinois.edu), 2021-01-09

# Standard library includes
import sys, time

# External library includes
import numpy as np
import mujoco_py

# Project includes
from .benchmark import Benchmark
from ..utils.data_generation import *
from .. import System
from ..tasks import Task
from ..costs import Cost

def viz_halfcheetah_traj(env, traj, repeat):
    for _ in range(repeat):
        env.reset()
        qpos = traj[0].obs[:9]
        qvel = traj[0].obs[9:]
        env.set_state(qpos, qvel)
        for i in range(len(traj)):
            u = traj[i].ctrl
            env.step(u)
            env.render()
            time.sleep(0.05)
        time.sleep(1)

def halfcheetah_dynamics(env, x, u, n_frames=5):
    old_state = env.sim.get_state()
    old_qpos = old_state[1]
    qpos = x[:len(old_qpos)]
    qvel = x[len(old_qpos):]
    new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
            old_state.act, old_state.udd_state)
    env.sim.set_state(new_state)
    #env.sim.forward()
    env.sim.data.ctrl[:] = u
    for _ in range(n_frames):
        env.sim.step()
    new_qpos = env.sim.data.qpos
    new_qvel = env.sim.data.qvel

    return np.concatenate([new_qpos, new_qvel])

class HalfcheetahCost(Cost):
    def __init__(self, env):
        self._is_quad = False
        self._is_convex = False
        self._is_diff = False
        self._is_twice_diff = False
        self._has_goal = False
        self.env = env
        self.fin_step = 1e-6

    def __call__(self, traj):
        cum_reward = 0.0
        for i in range(len(traj)-1):
            reward_ctrl = self.eval_ctrl_cost(traj[i].ctrl)
            reward_run = self.eval_obs_cost(traj[i].obs)
            cum_reward += reward_ctrl + reward_run
        
        cum_reward += self.eval_term_obs_cost(traj[-1].obs)
        return 200 - cum_reward

    def eval_obs_cost(self, obs):
        qpos, qvel = obs[0:9], obs[9:]

        # speed reward
        reward_velocity = qvel[0]

        # height velocity reward
        reward_height = -(0 - qvel[1])

        return 100 - (reward_velocity + reward_height)

    def eval_term_obs_cost(self, term_obs):
        qpos, qvel = term_obs[0:9], term_obs[9:]

        # speed reward
        reward_velocity = qvel[0]

        # height velocity reward
        reward_height = -(0 - qvel[1])

        return 100 - (reward_velocity + reward_height)

    def eval_ctrl_cost(self, ctrl):
        return 100 -0.1 * np.square(ctrl).sum()

    def eval_term_obs_cost_hess(self, term_obs):
        n = term_obs.shape[0]
        Jac = np.zeros(n)
        Hess = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1
            f1 = self.eval_term_obs_cost(term_obs + self.fin_step * ei)
            f2 = self.eval_term_obs_cost(term_obs - self.fin_step * ei)
            Jac[i] = (f1 - f2) / (2 * self.fin_step)
            
            for j in range(n):
                ej = np.zeros(n)
                ej[j] = 1
                f1 = self.eval_term_obs_cost(term_obs + self.fin_step * ei + self.fin_step * ej)
                f2 = self.eval_term_obs_cost(term_obs + self.fin_step * ei - self.fin_step * ej)
                f3 = self.eval_term_obs_cost(term_obs - self.fin_step * ei + self.fin_step * ej)
                f4 = self.eval_term_obs_cost(term_obs - self.fin_step * ei - self.fin_step * ej)
                Hess[i, j] = (f1 - f2 - f3 + f4) / (4 * self.fin_step * self.fin_step)

        cost = self.eval_term_obs_cost(term_obs)

        return cost, Jac, Hess

    def eval_obs_cost_hess(self, obs):
        n = obs.shape[0]
        Jac = np.zeros(n)
        Hess = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1
            f1 = self.eval_obs_cost(obs + self.fin_step * ei)
            f2 = self.eval_obs_cost(obs - self.fin_step * ei)
            Jac[i] = (f1 - f2) / (2 * self.fin_step)
            
            for j in range(n):
                ej = np.zeros(n)
                ej[j] = 1
                f1 = self.eval_obs_cost(obs + self.fin_step * ei + self.fin_step * ej)
                f2 = self.eval_obs_cost(obs + self.fin_step * ei - self.fin_step * ej)
                f3 = self.eval_obs_cost(obs - self.fin_step * ei + self.fin_step * ej)
                f4 = self.eval_obs_cost(obs - self.fin_step * ei - self.fin_step * ej)
                Hess[i, j] = (f1 - f2 - f3 + f4) / (4 * self.fin_step * self.fin_step)

        cost = self.eval_obs_cost(obs)

        return cost, Jac, Hess

    def eval_ctrl_cost_hess(self, ctrl):
        n = ctrl.shape[0]
        Jac = np.zeros(n)
        Hess = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1
            f1 = self.eval_ctrl_cost(ctrl + self.fin_step * ei)
            f2 = self.eval_ctrl_cost(ctrl - self.fin_step * ei)
            Jac[i] = (f1 - f2) / (2 * self.fin_step)
            
            for j in range(n):
                ej = np.zeros(n)
                ej[j] = 1
                f1 = self.eval_ctrl_cost(ctrl + self.fin_step * ei + self.fin_step * ej)
                f2 = self.eval_ctrl_cost(ctrl + self.fin_step * ei - self.fin_step * ej)
                f3 = self.eval_ctrl_cost(ctrl - self.fin_step * ei + self.fin_step * ej)
                f4 = self.eval_ctrl_cost(ctrl - self.fin_step * ei - self.fin_step * ej)
                Hess[i, j] = (f1 - f2 - f3 + f4) / (4 * self.fin_step * self.fin_step)

        cost = self.eval_ctrl_cost(ctrl)

        return cost, Jac, Hess

def gen_trajs(env, system, num_trajs=1000, traj_len=1000, seed=42):
    rng = np.random.default_rng(seed)
    trajs = []
    env.seed(int(rng.integers(1 << 30)))
    env.action_space.seed(int(rng.integers(1 << 30)))
    for i in range(num_trajs):
        init_obs = env.reset()
        traj = ampc.zeros(system, traj_len)
        traj[0].obs[:] = np.concatenate([[0], init_obs])
        for j in range(1, traj_len):
            action = env.action_space.sample()
            traj[j-1].ctrl[:] = action
            #obs, reward, done, info = env.step(action)
            obs = halfcheetah_dynamics(env, traj[j-1].obs[:], action)
            traj[j].obs[:] = obs
        trajs.append(traj)
    return trajs


class HalfcheetahBenchmark(Benchmark):
    """
    This benchmark uses the OpenAI gym halfcheetah benchmark and is consistent with the
    experiments in the ICRA 2021 paper. The benchmark requires OpenAI gym and mujoco_py
    to be installed.  The performance metric is
    :math:`200-R` where :math:`R` is the gym reward.
    """
    def __init__(self, data_gen_method="uniform_random"):
        name = "halfcheetah"
        system = ampc.System([f"x{i}" for i in range(18)], [f"u{i}" for i in range(6)])

        import gym, mujoco_py
        env = gym.make("HalfCheetah-v2")
        self.env = env

        system.dt = env.dt
        cost = HalfcheetahCost(env)
        task = Task(system)
        task.set_cost(cost)
        task.set_ctrl_bounds(env.action_space.low, env.action_space.high)
        init_obs = np.concatenate([env.init_qpos, env.init_qvel])
        task.set_init_obs(init_obs)
        task.set_num_steps(200)


        super().__init__(name, system, task, data_gen_method)

    def dynamics(self, x, u):
        return halfcheetah_dynamics(self.env,x,u)

    def gen_trajs(self, seed, n_trajs, traj_len=200):
        return gen_trajs(self.env, self.system, n_trajs, traj_len, seed)

    def visualize(self, traj, repeat):
        """
        Visualize the half-cheetah trajectory using Gym functions.

        Parameters
        ----------
        traj : Trajectory
            Trajectory to visualize

        repeat : int
            Number of times to repeat trajectory in visualization
        """
        viz_halfcheetah_traj(self.env, traj, repeat)

    @staticmethod
    def data_gen_methods():
        return ["uniform_random"]
