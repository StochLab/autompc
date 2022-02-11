import dill
file_path = "half_cheetah_sysid_params.pickle"
half_cheetah_sysid_params = dill.load(open(file_path, "rb"))

import autompc as ampc
import numpy as np
from autompc.benchmarks import HalfcheetahBenchmark

benchmark = HalfcheetahBenchmark()

system = benchmark.system
trajs = benchmark.gen_trajs(seed=10, n_trajs=3, traj_len=20)

from autompc.sysid import SINDy

import dill
file_path = "sindy_half_cheetah.pickle"
model: SINDy = dill.load(open(file_path, "rb"))

import matplotlib.pyplot as plt
from autompc.graphs.kstep_graph import KstepPredAccGraph

graph = KstepPredAccGraph(system, trajs, kmax=20, metric="abserror")
graph.add_model(model, "SINDy")

fig = plt.figure()
ax = fig.gca()
graph(fig, ax)
ax.set_title("Error of SINDy model on Half Cheetah")
plt.show()