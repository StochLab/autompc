import autompc as ampc
import numpy as np
from autompc.benchmarks import HalfcheetahBenchmark

benchmark = HalfcheetahBenchmark()

system = benchmark.system

# trajs = benchmark.gen_trajs(seed=100, n_trajs=3, traj_len=3)
# from autompc.sysid import SINDy
# model = SINDy(system, method="lstsq", trig_basis=True, trig_interaction=True)
# model.train(trajs)
# half_cheetah_sysid_params = model.model.get_params()
# print(half_cheetah_sysid_params)
# import dill
# with open("half_cheetah_sysid_params.pickle", "wb") as f:
    # dill.dump(half_cheetah_sysid_params, f)
# f.close()
# import dill
# with open("sindy_half_cheetah.pickle", "wb") as m:
    # dill.dump(model, m)
# m.close()

from autompc.sysid import GroundTruth
import gym
centroidal_env = gym.make('HalfCheetah-v2')
centroidal_env.reset()
masses = centroidal_env.model.body_mass
masses[2:] = 0
model = GroundTruth(system=system, env=centroidal_env)

from autompc.control import IterativeLQR
controller = IterativeLQR(system, benchmark.task, model, horizon=50, verbose=True)

# Create stub trajectory
traj = ampc.zeros(system, 1)
start = benchmark.task._init_obs
traj[0].obs[:] = start

# Initialize controller state
controller_state = controller.traj_to_state(traj)

traj = ampc.simulate(controller, init_obs=start, max_steps=750, 
                dynamics=benchmark.dynamics, whole_horizon=True)

import dill
with open("half_cheetah_traj_centroidal.pickle", "wb") as f:
    dill.dump(traj, f)
f.close()

# import dill
# with open("half_cheetah_traj.pickle", "rb") as f:
#     traj = dill.load(f)

benchmark.visualize(traj, repeat=1000)