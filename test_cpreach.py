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
from copy import deepcopy

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
def indicator_matrix_bounds(x):
    ndata = x.shape[0]
    dim = x.shape[1]
    
    
    # Discretize SS
    # Currently square grid space
    eps = 0.5
    lb = -4
    ub = 4
    
    # Define discrete intervals
    disc_intervals = []
    elems_arr = []
    for d in range(dim):
        this_interval = np.arange(lb, (ub + eps), eps)
        this_num_elems = this_interval.size - 1
        disc_intervals.append(this_interval)
        elems_arr.append(this_num_elems)
    num_grids = np.prod(elems_arr)
    
    # Create all possible intervals
    grids = []

    # Helper Function
    def helper(disc_intervals, depth):
        loc_grids = []
        if depth == 0:
            for i in range(len(disc_intervals[depth]) - 1):
                grid_pt = []
                grid_pt.append(disc_intervals[depth][i])
                grid_pt.append(disc_intervals[depth][i + 1])
                loc_grids.append(grid_pt)
        else:
            prev_grids = deepcopy(grids)
            for i in range(len(prev_grids)):
                for j in range(len(disc_intervals[depth]) - 1):
                    grid_pt = deepcopy(prev_grids[i])
                    grid_pt.append(disc_intervals[depth][j])
                    grid_pt.append(disc_intervals[depth][j + 1])
                    loc_grids.append(grid_pt)
        
        return loc_grids
    
    #
    for d in range(dim):
        grids = helper(disc_intervals, d)
    
    return grids


# 
def phi_fn(x, grids):
    x = np.array(x)
    
    
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    ndata = x.shape[0]
    dim = x.shape[1]
    
    # Initialize and fill in the indicator matrix
    if (ndata == 1):
        # print("Single Data Point")
        matrix = np.zeros((1, len(grids)))
        # print("\t", matrix.shape)
        for j, bounds in enumerate(grids):
            included = True
            for d in range(dim):
                if bounds[2*d] <= x.flatten()[d] < bounds[2*d + 1]:
                    continue
                else:
                    included = False
                    break
            if included:
                matrix[0, j] = 1
    else:
        # print("Multiple Data Points")
        matrix = np.zeros((ndata, len(grids)))
        # print("\t", matrix.shape)
        for i, val in enumerate(x):
            for j, bounds in enumerate(grids):
                included = True
                for d in range(dim):
                    if bounds[2*d] <= val[d] < bounds[2*d + 1]:
                        continue
                    else:
                        included = False
                        break
                if included:
                    matrix[i, j] = 1
    
    return matrix


#
def single_phi_fn(x, dim):
    eps = 0.5
    disc = np.arange(-4, 4 + eps, eps)
    
    x = np.array(x)
    
    if len(x.shape) == 1:
        x = x[dim]
    else:
        x = x[:, dim]

    # Create all possible intervals
    intervals = [(disc[i], disc[i + 1]) for i in range(len(disc) - 1)]

    # Initialize the indicator matrix
    matrix = np.zeros((len(x), len(intervals)))

    # Fill in the indicator matrix
    for i, value in enumerate(x):
        for j, (a, b) in enumerate(intervals):
            if a <= value < b:
                matrix[i, j] = 1

    return matrix

#
def ccp_setup(dkr, trainset):
    
    ## Pull Training Set Data
    x_ft = trainset['x'][:, 0]
    y_ft = trainset['x'][:, 1]
    n_trainset = x_ft.shape[0]
    dim = x_ft.shape[1]
    
    ## Add Disturbance to Calibration/Training Set
    x_cal = x_ft
    y_cal = y_ft
    unoise_bounds_cal = 0.15
    x_dist = np.random.uniform(-unoise_bounds_cal, unoise_bounds_cal, size=x_cal.shape)
    y_dist = np.random.uniform(-unoise_bounds_cal, unoise_bounds_cal, size=y_cal.shape)
    x_cal = x_cal + 0*x_dist
    y_cal = y_cal + 1*y_dist
    
    ## Cond Conf Object Creation
    cond_conf_arr = []
    for i in range(dim):
        def score_fn(x, y):
            res = y - dkr.predict(x, np.zeros((x.shape[0], 0)))
            return res[:, i]
        
        ind_matrix = indicator_matrix_bounds(x_cal)
        phi = lambda x: phi_fn(x, grids=ind_matrix)
        # phi = lambda x: single_phi_fn(x, i)
        inf_params = {}
        print(f"Setting up Conditional Conformal Prediction (CCP) Problem for Variable {i+1}...")
        this_cond_conf = CondConf(score_fn, phi, infinite_params=inf_params)
        this_cond_conf.setup_problem(x_cal, y_cal)
        print(f"\t-> Score Bounds: [{np.min(this_cond_conf.scores_calib)}, {np.max(this_cond_conf.scores_calib)}]")
        
        cond_conf_arr.append(deepcopy(this_cond_conf))
    
    ## Return Cond Conf Array
    return cond_conf_arr
    

#
def ccp_bounds(dkr, condconf_arr, test_point):
    bounds = []
    
    for i, cond_conf in enumerate(condconf_arr):
        # Functions
        def score_inv_fn_ub(s, x):
            y_pred = dkr.predict(x.flatten(), np.zeros(1))
            return [ -np.inf, (y_pred[i] + s)[0] ]
        def score_inv_fn_lb(s, x):
            y_pred = dkr.predict(x.flatten(), np.zeros(1))
            return [ (y_pred[i] + s)[0], np.inf ]
        # Lower Bound
        res = cond_conf.predict((cp_alpha/2), test_point, score_inv_fn_lb, exact=True, randomize=True)
        lb = res[0] - test_point[i]
        # Upper Bound
        res = cond_conf.predict((1 - (cp_alpha/2)), test_point, score_inv_fn_ub, exact=True, randomize=True)
        ub = res[1] - test_point[i]
        bounds.append((lb, ub))
    
    return bounds
        
    
# 
def ccp_test(dkr, trainset, valset, ex_name):
    
    ## Pull Validation Set Data
    x_val = valset['x'][:, 0]
    y_val = valset['x'][:, 1]
    n_val = x_val.shape[0]
    dim = x_val.shape[1]
    unoise_bounds_val = 0.15
    x_dist = np.random.uniform(-unoise_bounds_val, unoise_bounds_val, size=x_val.shape)
    y_dist = np.random.uniform(-unoise_bounds_val, unoise_bounds_val, size=y_val.shape)
    x_val = x_val + 0*x_dist
    y_val = y_val + 1*y_dist
    
    ## Setup Cond Conf Objects
    cc_arr = ccp_setup(dkr, trainset)
    
    ## Storage Variables
    lbs = np.zeros((n_val, dim))
    ubs = np.zeros((n_val, dim))
    
    ## Prediction
    true_coverage_count = 0
    dim_coverage_count = [0] * dim
    for i in tqdm(range(n_val), desc="Predicting Bounds", leave=False):
        this_x = x_val[i, :]
        this_y = y_val[i, :]
        this_diff = this_y - this_x
        this_bounds = ccp_bounds(dkr, cc_arr, this_x)
        
        dims_covered = 0
        for j in range(dim):
            this_lb = this_bounds[j][0]
            this_ub = this_bounds[j][1]
            if (this_diff[j] >= this_lb) and (this_diff[j] <= this_ub):
                dim_coverage_count[j] += 1
                dims_covered += 1
            lbs[i, j] = this_lb
            ubs[i, j] = this_ub
        if dims_covered == dim:
            true_coverage_count += 1
    print(f"True Coverage Rate: {(true_coverage_count / n_val) * 100:.2f}%")
    for j in range(dim):
        print(f" -> Dim {j+1} Coverage Rate: {(dim_coverage_count[j] / n_val) * 100:.2f}%")
    
    ## Process Data: y1 vs x1
    sort_order_x1 = np.argsort(x_val[:, 0])
    y1_hat = dkr.predict(x_val[sort_order_x1, :], np.zeros((x_val.shape[0], 0)))[:, 0]
    x1_s = x_val[sort_order_x1, 0]
    y1_s = y_val[sort_order_x1, 0]
    lb1 = lbs[sort_order_x1, 0]
    ub1 = ubs[sort_order_x1, 0]
    print(f"(LB, UB) for Var 1 contains INF: ({np.any(np.isinf(lb1))}, {np.any(np.isinf(ub1))})")
    
    ## Process Data: y2 vs x2
    if dim == 2:
        sort_order_x2 = np.argsort(x_val[:, 1])
        y2_hat = dkr.predict(x_val[sort_order_x2, :], np.zeros((x_val.shape[0], 0)))[:, 1]
        x2_s = x_val[sort_order_x2, 1]
        y2_s = y_val[sort_order_x2, 1]
        lb2 = lbs[sort_order_x2, 1]
        ub2 = ubs[sort_order_x2, 1]
        print(f"(LB, UB) for Var 2 contains INF: ({np.any(np.isinf(lb2))}, {np.any(np.isinf(ub2))})")
    
    ## Plot Data
    plot_diff = 1
    
    fig = plt.figure(dpi=300, figsize=(10, 10))
    if dim == 2:
        ax = fig.add_subplot(2, 1, 1)
    if dim == 1:
        ax = fig.add_subplot(1, 1, 1)
    # ax.plot(x1_s, y1_s, '.', alpha=0.2, color='b')
    ax.plot(x_val[:, 0], (y_val[:, 0] - plot_diff*x_val[:, 0]), '.', alpha=0.2, color='b')
    # ax.plot(x_cal[:, 0], y_cal[:, 0], '.', alpha=0.2, color='orangered')
    ax.plot(x1_s, (y1_hat - plot_diff*x1_s), lw=1, color='k', alpha=0.5)
    ax.plot(x1_s, lb1, lw=2, color='aquamarine')
    ax.plot(x1_s, ub1, lw=2, color='aquamarine')
    ax.fill_between(x1_s.flatten(), lb1, ub1, color='aquamarine', alpha=0.4)
    ax.set_xlabel("$x_1(k)$")
    if plot_diff == 1:
        ax.set_ylabel("$x_1(k+1)$ - $x_1(k)$")
    else:
        ax.set_ylabel("$x_1(k+1)$")
    ax.set_title("CCP: State 1")
    ax.grid(True)
    
    if dim == 2:
        ax = fig.add_subplot(2, 1, 2)
        # ax.plot(x2_s, y2_s, '.', alpha=0.2, color='b')
        ax.plot(x_val[:, 1], (y_val[:, 1] - plot_diff*x_val[:, 1]), '.', alpha=0.2, color='b')
        # ax.plot(x_cal[:, 1], y_cal[:, 1], '.', alpha=0.2, color='orangered')
        ax.plot(x2_s, (y2_hat - plot_diff*x2_s), lw=1, color='k', alpha=0.5)
        ax.plot(x2_s, lb2, lw=2, color='aquamarine')
        ax.plot(x2_s, ub2, lw=2, color='aquamarine')
        ax.fill_between(x2_s.flatten(), lb2, ub2, color='aquamarine', alpha=0.4)
        ax.set_xlabel("$x_2(k)$")
        if plot_diff == 1:
            ax.set_ylabel("$x_2(k+1)$ - $x_2(k)$")
        else:
            ax.set_ylabel("$x_2(k+1)$")
        ax.set_title("CCP: State 2")
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
    
    ## Calc
    
    return cc_arr


if __name__ == "__main__":
    
    start_time = time()

    # example = 'vanderpol'
    # example = 'brunton'
    example = 'duffing'
    # example = 'univariate'
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
