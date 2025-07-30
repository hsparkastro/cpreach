import yaml
import pickle
from tqdm import tqdm
from time import time
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import MultiPolygon
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from datetime import datetime

from depikt import Koopman
from depikt.utils.dataset import load_dataset
from reach import conformal_prediction, approximate_FRS, Zonotope
from reach import alphashape, polygon_to_halfspace
from reach.plot_helper import plot_rectangle
from reach.zonotope import Zonotope

# from condconf.conditionalconformal.synthetic_data import indicator_matrix
from condconf.conditionalconformal import CondConf


cp_alpha = 1 - 0.9
ashape_alpha = 0.0
area_threshold = 0.01
max_vertices = 20

np.set_printoptions(formatter={'float': '{: .3E}'.format})

""" helper functions """

# 
def phi_fn(x):
    x = np.array(x)
    
    # Discretize SS
    eps = 1
    disc_x = np.arange(-4, (4 + eps), eps)
    disc_y = np.arange(-4, (4 + eps), eps)
    
    # Create all possible intervals
    grids = []
    for i in range(len(disc_x) - 1):
        for j in range(len(disc_y) - 1):
            grids.append((disc_x[i], disc_x[i + 1], disc_y[j], disc_y[j + 1]))
    
    # Initialize and fill in the indicator matrix
    if x.shape[0] == 1 or x.ndim == 1:
        matrix = np.zeros((1, len(grids)))
        for j, (xlb, xub, ylb, yub) in enumerate(grids):
            if xlb <= x[0] < xub and ylb <= x[1] < yub:
                matrix[0, j] = 1
    else:
        matrix = np.zeros((x.shape[0], len(grids)))
        for i, val in enumerate(x):
            for j, (xlb, xub, ylb, yub) in enumerate(grids):
                if xlb <= val[0] < xub and ylb <= val[1] < yub:
                    matrix[i, j] = 1
    
    return matrix
    
    
# 
def ccp_test(dkr, trainset, valset, ex_name):
    
    ## Split Data Sets
    # split_ratio = 0.5
    x_ft = trainset['x'][:, 0]
    y_ft = trainset['x'][:, 1]
    # u = calset['u'][:, 0]
    
    n_trainset = ( x_ft.shape[0] )
    # n_train = int(split_ratio * n_trainset)
    # n_calib = n_trainset - n_train
    
    # x_train = x_ft[ : n_train ]
    # y_train = y_ft[ : n_train ]
    # x_cal = x_ft[ n_train : ]
    # y_cal = y_ft[ n_train : ]
    x_cal = x_ft
    y_cal = y_ft
    unoise_bounds = 0.25
    x_dist = np.random.uniform(-unoise_bounds, unoise_bounds, size=x_cal.shape)
    y_dist = np.random.uniform(-unoise_bounds, unoise_bounds, size=y_cal.shape)
    x_cal = x_cal + x_dist
    y_cal = y_cal + y_dist
        
    x_val = valset['x'][:, 0]
    y_val = valset['x'][:, 1]
    n_val = x_val.shape[0]
    x_dist = np.random.uniform(-unoise_bounds, unoise_bounds, size=x_val.shape)
    y_dist = np.random.uniform(-unoise_bounds, unoise_bounds, size=y_val.shape)
    x_val = x_val + x_dist
    y_val = y_val + y_dist
    
    ## Misc
    u_empty = np.zeros((x_ft.shape[0], 0))
    y_pred = dkr.predict(x_ft, u_empty)
    
    err = (y_ft - y_pred)
    mean = np.mean(err, axis=0)
    var = np.diag(np.cov(err.T))
    rmse = (mean**2 + var)**0.5
    maxerr = np.max(np.abs(err), axis=0)    
    
    ## Cond Conf Object Creation
    # score_fn = lambda x, y : np.linalg.norm(y - dkr.predict(x, np.zeros((x.shape[0], 0))), axis=1)
    # score_inv_fn_ub = lambda s, x : [ -np.inf, np.linalg.norm(dkr.predict(x, np.zeros((x.shape[0], 0))), axis=1) + s ]
    # score_inv_fn_lb = lambda s, x : [ np.linalg.norm(dkr.predict(x, np.zeros((x.shape[0], 0))), axis=1) + s, np.inf]
    score_scale = 0.5
    score_fn = lambda x, y : score_scale*np.sum(y - dkr.predict(x, np.zeros((x.shape[0], 0))), axis=1)
    # def score_fn(x, y):
    #     y_pred = dkr.predict(x, np.zeros((x.shape[0], 0)))
    #     res = np.sum(y - y_pred, axis=1)
    #     print("Residual Matrix Size: ", res.shape)
    #     return res
    score_inv_fn_ub = lambda s, x : [ -np.inf, score_scale*np.sum(dkr.predict(x, np.zeros((x.shape[0], 0))), axis=1) + s ]
    score_inv_fn_lb = lambda s, x : [ score_scale*np.sum(dkr.predict(x, np.zeros((x.shape[0], 0))), axis=1) + s, np.inf]
    
    phi = phi_fn
    inf_params = {}
    print("Setting up Conditional Conformal Prediction (CCP) Problem...")
    cond_conf = CondConf(score_fn, phi, infinite_params=inf_params)
    cond_conf.setup_problem(x_cal, y_cal)
    print(f"Score Bounds: [{np.min(cond_conf.scores_calib)}, {np.max(cond_conf.scores_calib)}]")
    
    ## Storage Variables
    lbs = np.zeros((n_val, x_val.shape[1]))
    ubs = np.zeros((n_val, x_val.shape[1]))
    
    ## Prediction
    for i in tqdm(range(n_val), desc="Predicting Bounds", leave=False):
        this_x = x_val[i, :]
        # print(this_x.reshape(-1, 1))
        res = cond_conf.predict((cp_alpha/2), this_x, score_inv_fn_lb, exact=True, randomize=True)
        # print(res)
        lbs[i] = res[0]
        res = cond_conf.predict((1 - (cp_alpha/2)), this_x, score_inv_fn_ub, exact=True, randomize=True)
        # print(res)
        ubs[i] = res[1]
    
    ## Process Data: y1 vs x1
    sort_order_x1 = np.argsort(x_val[:, 0])
    y1_hat = dkr.predict(x_val[sort_order_x1, :], np.zeros((x_val.shape[0], 0)))[:, 0]
    x1_s = x_val[sort_order_x1, 0]
    y1_s = y_val[sort_order_x1, 0]
    lb1 = lbs[sort_order_x1, 0]
    ub1 = ubs[sort_order_x1, 0]
    print(f"(LB, UB) for Var 1 contains INF: ({np.any(np.isinf(lb1))}, {np.any(np.isinf(ub1))})")
    
    ## Process Data: x1 vs x2
    sort_order_x2 = np.argsort(x_val[:, 1])
    y2_hat = dkr.predict(x_val[sort_order_x2, :], np.zeros((x_val.shape[0], 0)))[:, 1]
    x2_s = x_val[sort_order_x2, 1]
    y2_s = y_val[sort_order_x2, 1]
    lb2 = lbs[sort_order_x2, 1]
    ub2 = ubs[sort_order_x2, 1]
    print(f"(LB, UB) for Var 2 contains INF: ({np.any(np.isinf(lb2))}, {np.any(np.isinf(ub2))})")
    
    ## Plot Data
    fig = plt.figure(dpi=300, figsize=(10, 10))
    ax = fig.add_subplot(2, 1, 1)
    # ax.plot(x1_s, y1_s, '.', alpha=0.2, color='b')
    ax.plot(x_val[:, 0], y_val[:, 0], '.', alpha=0.2, color='b')
    # ax.plot(x_cal[:, 0], y_cal[:, 0], '.', alpha=0.2, color='orangered')
    ax.plot(x1_s, y1_hat, lw=1, color='k', alpha=0.5)
    ax.plot(x1_s, lb1, lw=2, color='aquamarine')
    ax.plot(x1_s, ub1, lw=2, color='aquamarine')
    ax.fill_between(x1_s.flatten(), lb1, ub1, color='aquamarine', alpha=0.4)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$y_1$")
    ax.set_title("CCP: $y_1$ vs $x_1$")
    ax.grid(True)
    
    ax = fig.add_subplot(2, 1, 2)
    # ax.plot(x2_s, y2_s, '.', alpha=0.2, color='b')
    ax.plot(x_val[:, 1], y_val[:, 1], '.', alpha=0.2, color='b')
    # ax.plot(x_cal[:, 1], y_cal[:, 1], '.', alpha=0.2, color='orangered')
    ax.plot(x2_s, y2_hat, lw=1, color='k', alpha=0.5)
    ax.plot(x2_s, lb2, lw=2, color='aquamarine')
    ax.plot(x2_s, ub2, lw=2, color='aquamarine')
    ax.fill_between(x2_s.flatten(), lb2, ub2, color='aquamarine', alpha=0.4)
    ax.set_xlabel("$x_2$")
    ax.set_ylabel("$y_2$")
    ax.set_title("CCP: $y_2$ vs $x_2$")
    ax.grid(True)
    
    ## Save Plot
    save_plt = True
    dt_string = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    fn_plt = f"figs/ccp-{ex_name}-{dt_string}.pdf"
    if save_plt:
        plt.savefig(fn_plt, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {fn_plt}")
        
    ## Show Plots
    # plt.show()
    
    return cond_conf


if __name__ == "__main__":
    
    start_time = time()

    example = 'vanderpol'
    # example = 'brunton'
    # example = 'duffing'
    directory = f'examples/{example}'
    config_file = 'config/standard.yaml'


    """ load parameters """
    
    with open(directory + '/' + config_file, 'r') as file:
        loaded = yaml.safe_load(file)
        model_args = loaded['model_args']
        fit_args = loaded['fit_args']


    """ load model """
    
    dkr = Koopman.from_args(**model_args)
    dkr.load(directory + '/' + fit_args['savepath'])


    """ conformal prediction """
    
    trainset = load_dataset(directory + '/' + fit_args['trainset'], numpy=True)
    # calset = load_dataset(directory + '/' + fit_args['trainset'], numpy=True)
    valset = load_dataset(directory + '/' + fit_args['valset'], numpy=True)
    cp_set = ccp_test(dkr, trainset, valset, example)
    print("Conformal Prediction completed in ", time() - start_time, "seconds")
    exit()

    """ calculate reachable set """
    
    T = 7
    dt = 0.01
    K = int(T / dt)
    t_eval = np.floor((np.arange(0, T+0.001, 1) / dt)).astype(int)
    nsamples_frs = 500
    dims = (0, 2)  # dimensions wanted in the reachable set
    
    
    """ set initial set (zonotope)"""
    
    initset = Zonotope(
        np.array([1.4, 2.4]),
        np.diag([0.15, 0.05])
    )

    def model(state, disturbance):
        empty_input = np.zeros((state.shape[0], 0))
        return dkr.predict(state, empty_input) + disturbance

    t0 = time()

    RFRS_trajs = approximate_FRS(
        initset, cp_set, model, T=K, nsamples=nsamples_frs,
        constant_input=False)
    t1 = time()

    RFRS_alphashape = []
    for t in t_eval:
        RFRS_samples = RFRS_trajs[t]
        ashape = alphashape(
            RFRS_samples, alpha=ashape_alpha)
        RFRS_alphashape.append(ashape)
    t2 = time()

    t3 = time()
    print('Elapsed time (FRS): ', t1-t0)
    print('Elapsed time (alphashape): ', t2-t1)
    print('Elapsed time (halfspace): ', t3-t2)


    """ plot """
    
    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    ashape = RFRS_alphashape[0]
    if isinstance(ashape, MultiPolygon):
        for ashape_i in ashape.geoms:
            x, y = ashape_i.exterior.xy
            ax.fill(x, y, edgecolor=[0, 1, 0],
                    facecolor=[0, 1, 0, 0.5], linestyle='-')
    else:
        x, y = ashape.exterior.xy
        ax.fill(x, y, edgecolor=[0, 1, 0],
                facecolor=[0, 1, 0, 0.5], linestyle='-')

    for i in range(1, len(t_eval)-1):

        ashape = RFRS_alphashape[i]

        if isinstance(ashape, MultiPolygon):
            for ashape_i in ashape.geoms:
                x, y = ashape_i.exterior.xy
                ax.fill(x, y, edgecolor=[1, 0, 0, 0.5],
                        facecolor=[1, 0, 0, 0.1], linestyle='-')
        else:
            x, y = ashape.exterior.xy
            ax.fill(x, y, edgecolor=[1, 0, 0, 0.5],
                    facecolor=[1, 0, 0, 0.1], linestyle='-')

    ashape = RFRS_alphashape[-1]
    if isinstance(ashape, MultiPolygon):
        for ashape_i in ashape.geoms:
            x, y = ashape_i.exterior.xy
            ax.fill(x, y, edgecolor=[0, 0, 1],
                    facecolor=[0, 0, 1, 0.5], linestyle='-')
    else:
        x, y = ashape.exterior.xy
        ax.fill(x, y, edgecolor=[0, 0, 1],
                facecolor=[0, 0, 1, 0.5], linestyle='-')

    ax.scatter(valset['x'][:, -1, 0], valset['x'][:, -1, 1],
               s=1, c='black', marker='o', label='Validation Set')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    # ax.legend(loc=2)
    plt.tight_layout()
    plt.show()
