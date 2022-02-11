# Standard library library
import sys

# Internal library includes
from .. import zeros, extend

# External library includes
import numpy as np
from tqdm import tqdm

def simulate(controller, init_obs, term_cond=None, dynamics=None, sim_model=None, max_steps=10000, ctrl_bounds=None, silent=False, whole_horizon=False):
    """
    Simulate a controller with respect to a dynamics function or simulation model.

    Parameters
    ----------
    controller : Controller
        Controller to simulate

    init_obs : numpy array of size controller.system.obs_dim
        Initial observation

    term_cond : Function Trajectory -> bool
        Function which returns true when termination condition is met.

    dynamics : Function obs, control -> newobs
        Function defining system dynamics

    sim_model : Model
        Simulation model.  Used when dynamics is None

    max_steps : int
        Maximum number of simulation steps allowed.  Default is 10000.

    silent : bool
        Suppress output if True.
    """
    if dynamics is None and sim_model is None:
        raise ValueError("Must specify dynamics function or simulation model")

    sim_traj = zeros(controller.system, 1)
    x = np.copy(init_obs)
    sim_traj[0].obs[:] = x
    
    constate = controller.traj_to_state(sim_traj)
    if dynamics is None:
        simstate = sim_model.traj_to_state(sim_traj)

    if whole_horizon:
        if silent:
            itr = range(int(max_steps // controller.horizon))
        else:
            itr = tqdm(range(int(max_steps // controller.horizon)), file=sys.stdout)

        for _  in itr:
            u, constate = controller.run(constate, sim_traj[-1].obs, whole_horizon=whole_horizon)
            for i in range(controller.horizon):
                if ctrl_bounds is not None:
                    u[i] = np.clip(u[i], ctrl_bounds[:,0], ctrl_bounds[:,1])
                if dynamics is None:
                    simstate = sim_model.pred(simstate, u[i])
                    x = simstate[:controller.system.obs_dim]
                else:
                    x = dynamics(x, u[i])
                sim_traj[-1].ctrl[:] = u[i]
                sim_traj = extend(sim_traj, [x], 
                        np.zeros((1, controller.system.ctrl_dim)))
                if term_cond is not None and term_cond(sim_traj):
                    break
    else:
        if silent:
            itr = range(max_steps)
        else:
            itr = tqdm(range(max_steps), file=sys.stdout)

        for _  in itr:
            u, constate = controller.run(constate, sim_traj[-1].obs)
            if ctrl_bounds is not None:
                u = np.clip(u, ctrl_bounds[:,0], ctrl_bounds[:,1])
            if dynamics is None:
                simstate = sim_model.pred(simstate, u)
                x = simstate[:controller.system.obs_dim]
            else:
                x = dynamics(x, u)
            sim_traj[-1].ctrl[:] = u
            sim_traj = extend(sim_traj, [x], 
                    np.zeros((1, controller.system.ctrl_dim)))
            if term_cond is not None and term_cond(sim_traj):
                break

    return sim_traj
