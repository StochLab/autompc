import autompc as ampc
import numpy as np
from autompc.benchmarks import HalfcheetahBenchmark

benchmark = HalfcheetahBenchmark()

system = benchmark.system
trajs = benchmark.gen_trajs(seed=100, n_trajs=4, traj_len=4)

from autompc.sysid import SINDy

model = SINDy(system, method="lstsq", trig_basis=True, trig_interaction=True)
model.train(trajs)

from autompc.control import IterativeLQR
controller = IterativeLQR(system, benchmark.task, model, horizon=10)

# Create stub trajectory
traj = ampc.zeros(system, 1)
start = benchmark.task._init_obs
traj[0].obs[:] = start

# Initialize controller state
controller_state = controller.traj_to_state(traj)

traj = ampc.simulate(controller, init_obs=start, max_steps=200, dynamics=benchmark.dynamics)

benchmark.visualize(traj, repeat=1000)