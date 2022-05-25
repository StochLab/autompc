import numpy as np
import numba
import time

##################################
######### iLQR params ############
##################################

spec = [
    ('OVERFLOW_CAP', numba.float64),
    ('FIN_DIFF_STEP_SIZE', numba.float64),
    ('DT', numba.float64),

    ('X_START', numba.float64[:]),
    ('X_GOAL', numba.float64[:]),
    ('U_NOMINAL', numba.float64[:]),
    ('X_DIM', numba.int64),
    ('U_DIM', numba.int64),
    ('OBS_DIM', numba.int64),
    
    ('ROBOT_RADIUS', numba.float64),
    ('WHEEL_RADIUS', numba.float64),
    ('WHEEL_DIST', numba.float64),
    ('BOTTOM_LEFT', numba.float64[:]),
    ('TOP_RIGHT', numba.float64[:]),    

    ('BOUNDARY_FACTOR', numba.float64),
    ('Q', numba.float64[:, :]),
    ('R', numba.float64[:, :]),
    ('Qf', numba.float64[:, :]),
    ('ROT_COST', numba.float64),

    ('HORIZON_LENGTH', numba.int64),
    ('MAX_ITER', numba.int64),
    ('TOL', numba.float64),    
]

@numba.experimental.jitclass(spec)
class iLQR_params(object):
    def __init__(self, xdim, udim):
        self.OVERFLOW_CAP = 1e5
        self.FIN_DIFF_STEP_SIZE = 1e-4
        self.DT = 1/6

        self.X_DIM = xdim
        self.U_DIM = udim
        self.X_START = np.zeros(self.X_DIM)

        self.Q = 30.*np.eye(self.X_DIM)
        self.R = 0.6*np.eye(self.U_DIM)
        self.Qf = 60.*np.eye(self.X_DIM)
        self.ROT_COST = 0.4

        self.HORIZON_LENGTH = 1
        self.MAX_ITER = 1
        self.TOL = 1e-1

##################################
######### UTILS FUNCTIONS ########
##################################

@numba.njit(fastmath=True, parallel=False)
def regularize(Q):
    w, v = np.linalg.eig(Q)

    for i in numba.prange(w.shape[0]):
        if w[i] < 1e-6:
            w[i] = 1e-6

    Q_new = np.real(v.dot(np.diag(w).dot(v.T)))
    return Q_new

###############################################################################
###################### TURTLEBOT DIFFDRIVE DYNAMICS ###########################
###############################################################################

@numba.njit(fastmath=True, parallel=False)
def f(x, u, params: iLQR_params):
    '''
    Continuous time dynamics
    '''
    raise NotImplementedError

@numba.njit(fastmath=True, parallel=False)
def g(x, u, params: iLQR_params):
    '''
    Discrete time dynamics
    '''
    k1 = f(x, u, params)
    k2 = f(x + 0.5*params.DT*k1, u, params)
    k3 = f(x + 0.5*params.DT*k2, u, params)
    k4 = f(x + params.DT*k3, u, params)

    return x + (params.DT/6.0)*(k1 + 2*k2 + 2*k3 + k4)

@numba.njit(fastmath=True, parallel=False)
def finite_diff_jacobian_x(x, u, func, params: iLQR_params):
    A = np.zeros((params.X_DIM, params.X_DIM))
    ar = np.copy(x)
    al = np.copy(x)

    for i in numba.prange(params.X_DIM):
        ar[i] += params.FIN_DIFF_STEP_SIZE
        al[i] -= params.FIN_DIFF_STEP_SIZE
        A[:, i] = (func(ar, u, params) - func(al, u, params)) / (2*params.FIN_DIFF_STEP_SIZE)
        ar[i] = al[i] = x[i]

    return A

@numba.njit(fastmath=True, parallel=False)
def finite_diff_jacobian_u(x, u, func, params: iLQR_params):

    B = np.zeros((params.X_DIM, params.U_DIM))
    br = np.copy(u)
    bl = np.copy(u)

    for i in numba.prange(params.U_DIM):
        br[i] += params.FIN_DIFF_STEP_SIZE
        bl[i] -= params.FIN_DIFF_STEP_SIZE
        B[:, i] = (func(x, br, params) - func(x, bl, params)) / (2*params.FIN_DIFF_STEP_SIZE)
        br[i] = bl[i] = u[i]

    return B

###############################################################################
############################# TURTLEBOT COST ##################################
###############################################################################

@numba.njit(fastmath=True, parallel=False)
def cost_fun(x, u, t, params: iLQR_params):
    '''
    Cost function
    '''
    raise NotImplementedError

@numba.njit(fastmath=True, parallel=False)
def final_cost_fun(x, params: iLQR_params):
    '''
    Final cost function
    '''
    raise NotImplementedError

@numba.njit(fastmath=True, parallel=False)
def get_state_action_cost(xs, us, params: iLQR_params):
    cost = 0.0
    ind = 0
    for ind in numba.prange(len(us)):
        cost += cost_fun(xs[ind], us[ind], ind, params)
    cost += final_cost_fun(xs[-1], params)
    return cost

@numba.njit(fastmath=True, parallel=False)
def get_cost(xs, us, params: iLQR_params):
    return get_state_action_cost(xs, us, params)

@numba.njit(fastmath=True, parallel=False)
def quadratize_cost(x, u, t, it, params: iLQR_params):
    '''
    Quadratizes the cost around (x, u)
    '''
    raise NotImplementedError

@numba.njit(fastmath=True, parallel=False)
def quadratize_final_cost(x, params: iLQR_params):
    '''
    Quadratizes the final cost around x
    '''
    raise NotImplementedError

###############################################################################
############################# ILQR FUNCTIONs ##################################
###############################################################################

@numba.njit(fastmath=True, parallel=False)
def rollout(init_x, l, L, params: iLQR_params, verbose=False):
    xs = numba.typed.List()
    us = numba.typed.List()
    x = init_x
    xs.append(x)
    for t in numba.prange(params.HORIZON_LENGTH):
        if verbose:
            print(t, ': ', x)
        u = L[t].dot(x) + l[t]
        x = g(x, u, params)

        xs.append(x)
        us.append(u)
    if verbose:
        print(t, ': ', x)

    return xs, us

@numba.njit(fastmath=True, parallel=False)
def ilqr(init_x, l, L, params: iLQR_params, verbose=False):

    costs = []

    if params.MAX_ITER is None:
        params.MAX_ITER = 1000

    if L is None or l is None:
        L = numba.typed.List()
        l = numba.typed.List()
        for _ in numba.prange(params.HORIZON_LENGTH):
            L.append(np.zeros((params.U_DIM, params.X_DIM)))
            l.append(np.zeros(params.U_DIM))
    else:
        assert len(L) == params.HORIZON_LENGTH            

    # Initialize xhat, uhat with current control
    xHat, uHat = rollout(init_x, l, L, params, verbose=False)

    xHatNew = numba.typed.List()
    uHatNew = numba.typed.List()
    for _ in numba.prange(params.HORIZON_LENGTH):
        xHatNew.append(np.zeros(params.X_DIM))
        uHatNew.append(np.zeros(params.U_DIM))
    xHatNew.append(np.zeros(params.X_DIM)) # H + 1

    oldCost = np.inf

    for it in numba.prange(params.MAX_ITER):
        alpha = 1.0
        sub_iter = 0
        while True:
            newCost = 0.0
            # for indx in numba.prange(X_DIM):
            #     xHatNew[0][indx] = X_START[indx]
            xHatNew[0] = init_x
            for t in numba.prange(params.HORIZON_LENGTH):
                uHatNew[t] = (1.0 - alpha)*uHat[t] + \
                    L[t].dot(xHatNew[t] - (1.0 - alpha)*xHat[t]) + alpha*l[t]
                xHatNew[t+1] = g(xHatNew[t], uHatNew[t], params)
                newCost += cost_fun(xHatNew[t], uHatNew[t], t, params)

            newCost += final_cost_fun(xHatNew[params.HORIZON_LENGTH], params)
            rel_progress = np.abs((oldCost - newCost) / newCost)
            if (newCost < oldCost) or (rel_progress < params.TOL) or (sub_iter > 2):
                break
            alpha *= 0.5
            sub_iter += 1

        xHat = xHatNew
        uHat = uHatNew
        costs.append(get_cost(xHat, uHat, params))

        if verbose:
            print("Iter : ", it, " alpha : ", alpha,
                  " rel. progress : ",rel_progress," cost : ",newCost)

        if (rel_progress < params.TOL) and (params.MAX_ITER > 5):
            return (it, l, L, costs)

        oldCost = newCost

        s, S = quadratize_final_cost(xHat[params.HORIZON_LENGTH], params)

        for t in numba.prange(params.HORIZON_LENGTH-1, -1, -1):
            A = finite_diff_jacobian_x(xHat[t], uHat[t], g, params)
            B = finite_diff_jacobian_u(xHat[t], uHat[t], g, params)

            c = xHat[t+1] - A.dot(xHat[t]) - B.dot(uHat[t])

            P, q, Q, r, R = quadratize_cost(xHat[t], uHat[t], t, it, params)

            C = B.T.dot(S.dot(A)) + P
            D = A.T.dot(S.dot(A)) + Q
            E = B.T.dot(S.dot(B)) + R
            d = A.T.dot(s + S.dot(c)) + q
            e = B.T.dot(s + S.dot(c)) + r

            L[t] = -1*np.linalg.lstsq(E, C)[0]
            l[t] = -1*np.linalg.lstsq(E, e)[0]

            S = D + C.T.dot(L[t])
            s = d + C.T.dot(l[t])

    return (it, l, L, costs)

import copy

def compile():
    params = iLQR_params()
    # Reinitialize the linear policy
    l = numba.typed.List()
    L = numba.typed.List()
    for _ in numba.prange(params.HORIZON_LENGTH):
        l.append(np.zeros(params.U_DIM))
        L.append(np.zeros((params.U_DIM, params.X_DIM)))

    init_x = copy.deepcopy(params.X_START)
    # run ilqr without augmentation
    it, l, L, ilqr_total_costs = ilqr(init_x, l, L, params, verbose=False)
    ilqr_xs, ilqr_us = rollout(init_x, l, L, params, verbose=False)

############## PLOTTING #################
import matplotlib.pyplot as plt
# Plot the trajectory

def plot(path, ilqr_us, ilqr_total_costs, params: iLQR_params, ref_path=None):
    plt.subplot(1, 1, 1)
    # Get bounds of the environment
    xaxis = np.arange(params.BOTTOM_LEFT[0], params.TOP_RIGHT[0], 0.05)
    yaxis = np.arange(params.BOTTOM_LEFT[1], params.TOP_RIGHT[1], 0.05)
    # Set plt limits
    plt.xlim([xaxis[0], xaxis[-1]])
    plt.ylim([yaxis[0], yaxis[-1]])
    path_np = np.array(path)
    path_np = path_np[:, :2]  # Get only x, y
    ax = plt.gca()
    T = len(path) * 2.0
    for t in range(len(path)):
        circle = plt.Circle(path_np[t, :], params.ROBOT_RADIUS, color='r', alpha=(t+1)/T)
        ax.add_artist(circle)

    if ref_path is not None:
        plt.plot(ref_path[0], ref_path[1], '-o')

    plt.plot(params.X_START[0], params.X_START[1], marker='D', color='black')
    plt.plot(params.X_GOAL[0], params.X_GOAL[1], marker='D', color='green')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plot of robot path')

    # plt.subplot(2, 3, 1)
    # us = np.array(ilqr_us)
    # plt.plot(np.linspace(0,params.DT,len(ilqr_us)), params.WHEEL_RADIUS * (us[:, 0] + us[:, 1]), label='speed')
    # plt.legend()
    # plt.subplot(2, 3, 2)
    # plt.plot(ilqr_total_costs, label='cost trace')
    # plt.legend()
    # plt.subplot(2, 3, 3)
    # plt.plot(np.linspace(0,params.DT,len(ilqr_us)), (-us[:, 0] + us[:, 1])/params.WHEEL_DIST, label='omega')
    # plt.legend()


if __name__ == '__main__':

    compile()

    params = iLQR_params()
    params.MAX_ITER = 200
    params.TOL = 1e-3
    params.HORIZON_LENGTH = 150

    start = time.time()
    # Reinitialize the linear policy
    l = numba.typed.List()
    L = numba.typed.List()
    for _ in numba.prange(params.HORIZON_LENGTH):
        l.append(np.zeros(params.U_DIM))
        L.append(np.zeros((params.U_DIM, params.X_DIM)))

    init_x = copy.deepcopy(params.X_START)
    # run ilqr without augmentation
    it, l, L, ilqr_total_costs = ilqr(init_x, l, L, params, verbose=False)
    ilqr_xs, ilqr_us = rollout(init_x, l, L, params, verbose=False)
    
    end = time.time()
    print('ilqr Num iter : '+str(it) + ' true cost: '+str(ilqr_total_costs[-1]) +
          ' time : '+str(end - start))

    plt.gcf().set_size_inches([11.16, 7.26])
    plot(ilqr_xs, ilqr_us, ilqr_total_costs, params)
    plt.show()