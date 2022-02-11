# Mujoco Half-Cheetah IK Solver

import mujoco_py
import numpy as np
    
if __name__ == '__main__':
    import gym
    env = gym.make('HalfCheetah-v2')
    env.reset()
    masses = env.model.body_mass
    print(masses)
    masses[2:] = 0
    print(masses)
    for _ in range(1000):
        # env.step(env.action_space.sample())
        env.step(np.zeros(env.action_space.sample().shape))
        env.render()

