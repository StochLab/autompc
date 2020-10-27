from pdb import set_trace
import sys, os
sys.path.append(os.getcwd() + "/..")

import numpy as np
import autompc as ampc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from joblib import Memory

from scipy.integrate import solve_ivp

memory = Memory("cache")

cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])

def cartpole_dynamics(y, u, g = 9.8, m_c = 1, m_p = 1, L = 1, b = 1.0):
    """
    Parameters
    ----------
        y : states
        u : control

    Returns
    -------
        A list describing the dynamics of the cart cart pole
    """
    theta, omega, x, dx = y
    return np.array([omega,
            1.0/(L*(m_c+m_p+m_p*np.sin(theta)**2))*(-u*np.cos(theta) 
                - m_p*L*omega**2*np.cos(theta)*np.sin(theta)
                - (m_c+m_p+m_p)*g*np.sin(theta)
                - b*omega),
            dx,
            1.0/(m_c + m_p*np.sin(theta)**2)*(u + m_p*np.sin(theta)*
                (L*omega**2 + g*np.cos(theta)))])

def cartpole_simp_dynamics(y, u, g = 9.8, m = 1, L = 1, b = 0.1):
    """
    Parameters
    ----------
        y : states
        u : control

    Returns
    -------
        A list describing the dynamics of the cart cart pole
    """
    theta, omega, x, dx = y
    return np.array([omega,
            g * np.sin(theta)/L - b * omega / (m*L**2) + u * np.cos(theta)/L,
            dx,
            u])

def dt_cartpole_dynamics(y,u,dt,g=9.8,m=1,L=1,b=1.0):
    y = np.copy(y)
    #y[0] += np.pi
    #sol = solve_ivp(lambda t, y: cartpole_dynamics(y, u, g, m, L, b), (0, dt), y, t_eval = [dt])
    #if not sol.success:
    #    raise Exception("Integration failed due to {}".format(sol.message))
    #y = sol.y.reshape((4,))
    y += dt * cartpole_simp_dynamics(y,u[0],g,m,L,b)
    #y[0] -= np.pi
    return y

def animate_cartpole(fig, ax, dt, traj):
    ax.grid()

    line, = ax.plot([0.0, 0.0], [0.0, -1.0], 'o-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ctrl_text = ax.text(0.7, 0.95, '', transform=ax.transAxes)

    def init():
        line.set_data([0.0, 0.0], [0.0, -1.0])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        #i = min(i, ts.shape[0])
        line.set_data([traj[i,"x"], traj[i,"x"]+np.sin(traj[i,"theta"]+np.pi)], 
                [0, -np.cos(traj[i,"theta"] + np.pi)])
        time_text.set_text('t={:.2f}'.format(dt*i))
        ctrl_text.set_text("u={:.2f}".format(traj[i,"u"]))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=traj.size, interval=dt*1000,
            blit=False, init_func=init, repeat_delay=1000)

    return ani

dt = 0.05
cartpole.dt = dt

umin = -5.0
umax = 5.0

from cartpole_model import CartpoleModel
from autompc.control import FiniteHorizonLQR
from autompc.sysid.dummy_linear import DummyLinear

def get_generation_controller():
    truedyn = CartpoleModel(cartpole)
    _, A, B = truedyn.pred_diff(np.zeros(4,), np.zeros(1))
    model = DummyLinear(cartpole, A, B)
    Q = np.eye(4)
    R = 0.01 * np.eye(1)

    from autompc.tasks.quad_cost import QuadCost
    cost = QuadCost(cartpole, Q, R)

    from autompc.tasks.task import Task

    task = Task(cartpole)
    task.set_cost(cost)
    task.set_ctrl_bound("u", -20.0, 20.0)
    cs = FiniteHorizonLQR.get_configuration_space(cartpole, task, model)
    cfg = cs.get_default_configuration()
    cfg["horizon"] = 1000
    con = ampc.make_controller(cartpole, task, model, FiniteHorizonLQR, cfg)
    return con

# Generate trajectories for training
num_trajs = 500

@memory.cache
def gen_trajs(traj_len, num_trajs=num_trajs, dt=dt, rand_contr_prob=1.0):
    rng = np.random.default_rng(49)
    trajs = []
    con = get_generation_controller()
    for _ in range(num_trajs):
        theta0 = rng.uniform(-1.0, 1.0, 1)[0]
        y = [theta0, 0.0, 0.0, 0.0]
        traj = ampc.zeros(cartpole, traj_len)
        traj.obs[:] = y
        if rng.random() < rand_contr_prob:
            actuate = False
        else:
            actuate = True
            constate = con.traj_to_state(traj[:1])
        for i in range(traj_len):
            traj[i].obs[:] = y
            #if u[0] > umax:
            #    u[0] = umax
            #if u[0] < umin:
            #    u[0] = umin
            #u += rng.uniform(-udmax, udmax, 1)
            if not actuate:
                u  = rng.uniform(umin, umax, 1)
            else:
                u, constate = con.run(constate, y)
            y = dt_cartpole_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs
#trajs = gen_trajs(10)
#trajs2 = gen_trajs(200)
trajs3 = gen_trajs(200, rand_contr_prob=0.5)

from autompc.sysid import (ARX, Koopman, SINDy, 
        GaussianProcess, LargeGaussianProcess, 
        ApproximateGaussianProcess)

def train_gp(datasize=50, num_trajs=1):
    cs = GaussianProcess.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    gp = ampc.make_model(cartpole, GaussianProcess, cfg)
    mytrajs = [trajs2[i][:datasize] for i in range(num_trajs)]
    gp.train(mytrajs)
    return gp

def train_large_gp(datasize=50, num_trajs=1):
    cs = LargeGaussianProcess.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    gp = ampc.make_model(cartpole, LargeGaussianProcess, cfg)
    mytrajs = [trajs2[i][:datasize] for i in range(num_trajs)]
    gp.train(mytrajs)
    return gp

@memory.cache
def train_arx(k=2):
    cs = ARX.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    cfg["history"] = k
    arx = ampc.make_model(cartpole, ARX, cfg)
    arx.train(trajs)
    return arx

#@memory.cache
def train_koop():
    cs = Koopman.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    cfg["trig_basis"] = "true"
    cfg["poly_basis"] = "false"
    cfg["method"] = "lstsq"
    koop = ampc.make_model(cartpole, Koopman, cfg)
    koop.train(trajs2[:50])
    return koop

def train_sindy():
    cs = SINDy.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    cfg["trig_basis"] = "true"
    cfg["trig_freq"] = 1
    cfg["poly_basis"] = "false"
    sindy = ampc.make_model(cartpole, SINDy, cfg)
    sindy.train(trajs2)
    return sindy

def fd_jac(func, x, dt=1e-4):
    res = func(x)
    jac = np.empty((res.size, x.size))
    for i in range(x.size):
        xp = np.copy(x)
        xp[i] += dt
        resp = func(xp)
        jac[:,i] = (resp - res) / dt
    return jac

#@memory.cache
def train_approx_gp_inner(num_trajs):
    cs = ApproximateGaussianProcess.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    model = ampc.make_model(cartpole, ApproximateGaussianProcess, cfg)
    model.train(trajs3[-num_trajs:])
    return model.get_parameters()

def train_approx_gp(num_trajs):
    cs = ApproximateGaussianProcess.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    model = ampc.make_model(cartpole, ApproximateGaussianProcess, cfg)
    params = train_approx_gp_inner(num_trajs)
    model.set_parameters(params)
    return model

gp = train_approx_gp(25)
set_trace()
#import timeit
#m = 50
##t1 = timeit.Timer(lambda: gp.pred_parallel(np.zeros((m,4)), np.ones((m,1))))
##res1 = gp.pred_diff_parallel(np.zeros((m,4)), np.ones((m,1)))
##res2 = gp.pred_diff(np.zeros((4)), np.ones((1)))
##set_trace()
##gp.pred_timeit(np.zeros(4,), np.ones(1,))
#t1 = timeit.Timer(lambda: gp.pred(np.zeros(4,), np.ones(1,)))
#t2 = timeit.Timer(lambda: gp.pred_diff_parallel(np.zeros((m,4)), np.ones((m,1))))
#
#n = 100
#print(f"{t1.timeit(number=n)/n*1000=} ms")
#rep = t1.repeat(number=n)
#print("rep1=", [str(r/n*1000) + " ms" for r in rep])
#print(f"{t2.timeit(number=n)/n*1000=} ms")
#rep2 = t2.repeat(number=n)
#print("rep2=", [str(r/n*1000) + " ms" for r in rep2])
#
#sys.exit(0)


#arx = train_arx(k=4)
#koop = train_koop()
#sindy = train_sindy()

# Check inference time
## Training GP model
#import timeit
#sizes = []
#times = []
#for datasize in range(10, 60, 10):
#    num_trajs = 10
#    #gp = train_large_gp(datasize=datasize, num_trajs=num_trajs)
#    cs = LargeGaussianProcess.get_configuration_space(cartpole)
#    cfg = cs.get_default_configuration()
#    gp = ampc.make_model(cartpole, LargeGaussianProcess, cfg)
#    gp.train(trajs2[-5:])
#    state = np.zeros(4,)
#    state[0] = 0.5
#    x, state_jac, ctrl_jac = gp.pred_diff(state, np.ones(1,))
#    state_jac2 = fd_jac(lambda y: gp.pred(y, np.ones(1,)), state, dt=1e-2)
#    state_jac3 = fd_jac(lambda y: gp.pred(y, np.ones(1,)), state, dt=1e-3)
#    state_jac4 = fd_jac(lambda y: gp.pred(y, np.ones(1,)), state, dt=1e-4)
#    state_jac5 = fd_jac(lambda y: gp.pred(y, np.ones(1,)), state, dt=1e-5)
#    ctrl_jac2 = fd_jac(lambda y: gp.pred(state, y), np.ones(1,))
#    set_trace()
#    t = timeit.Timer(lambda: gp.pred(np.zeros(4,), np.ones(1,)))
#    print(t.timeit(number=1))
#    rep = t.repeat(number=3)
#    print(rep)
#    sizes.append(datasize*num_trajs)
#    times.append(rep[-1]*1000.0)
#plt.figure()
#ax = plt.gca()
#ax.plot(sizes, times)
#ax.set_xlabel("Training data points.")
#ax.set_ylabel("Inference time (ms)")
#plt.show()
#sys.exit(0)

if True:
    from autompc.evaluators import HoldoutEvaluator, FixedSetEvaluator
    from autompc.metrics import RmseKstepMetric
    from autompc.graphs import KstepGrapher, InteractiveEvalGrapher

    metric = RmseKstepMetric(cartpole, k=10)
    grapher = InteractiveEvalGrapher(cartpole, logscale=True)
    grapher2 = KstepGrapher(cartpole, kmax=50, kstep=5, evalstep=10,
            logscale=True)

    rng = np.random.default_rng(42)
    evaluator = FixedSetEvaluator(cartpole, trajs3[:20], metric, rng, 
            #training_trajs=trajs2[-5:]) 
            #training_trajs=[trajs2[0][:100], trajs2[3][150:200]]) 
            training_trajs=trajs3[-50:]) 
    evaluator.add_grapher(grapher)
    evaluator.add_grapher(grapher2)
    #cs = LargeGaussianProcess.get_configuration_space(cartpole)
    cs = ApproximateGaussianProcess.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    cfg["induce_count"] = 500
    #cfg["trig_basis"] = "true"
    #cfg["poly_basis"] = "false"
    #cfg["poly_degree"] = 3
    #cs = MyLinear.get_configuration_space(cartpole)
    #cfg = cs.get_default_configuration()
    eval_score, _, graphs = evaluator(ApproximateGaussianProcess, cfg)
    print("eval_score = {}".format(eval_score))
    fig = plt.figure()
    graph = graphs[0]
    graph.set_obs_lower_bound("theta", -0.2)
    graph.set_obs_upper_bound("theta", 0.2)
    graph.set_obs_lower_bound("omega", -0.2)
    graph.set_obs_upper_bound("omega", 0.2)
    graph.set_obs_lower_bound("dx", -0.2)
    graph.set_obs_upper_bound("dx", 0.2)
    graph.set_obs_lower_bound("x", -0.2)
    graph.set_obs_upper_bound("x", 0.2)
    graphs[0](fig)
    graphs[1](fig)
    #plt.tight_layout()
    plt.show()
    sys.exit(0)