import yaml
import pickle
from tqdm import trange
from time import time
import numpy as np
import matplotlib.pyplot as plt

from depikt import Koopman
from depikt.utils.dataset import load_dataset
from reach import conformal_prediction, approximate_FRS, Zonotope
from reach import alphashape, polygon_to_halfspace
from reach.plot_helper import plot_rectangle
from reach.zonotope import Zonotope
from shapely.geometry import MultiPolygon


cp_alpha = 1 - 0.9
ashape_alpha = 0.0
area_threshold = 0.01
max_vertices = 20

np.set_printoptions(formatter={'float': '{: .3E}'.format})


def comformal_prediction(dkr, calset):
    """ conformal prediction """
    x = calset['x'][:, 0]
    y = calset['x'][:, 1]
    # u = calset['u'][:, 0]
    u_empty = np.zeros((x.shape[0], 0))
    y_pred = dkr.predict(x, u_empty)
    err = (y - y_pred)
    mean = np.mean(err, axis=0)
    var = np.diag(np.cov(err.T))
    rmse = (mean**2 + var)**0.5
    maxerr = np.max(np.abs(err), axis=0)
    cp_bound = conformal_prediction(err, cp_alpha)

    print('CP Bound = ', cp_bound)
    print('RMS Err. = ', rmse)
    print('Max Err. = ', maxerr)

    cp_set = Zonotope(
        np.zeros(2),
        np.diag(cp_bound)
    )
    return cp_set


if __name__ == "__main__":

    directory = 'examples/vanderpol'
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
    calset = load_dataset(directory + '/' + fit_args['valset'], numpy=True)
    # calset = load_dataset(directory + '/' + fit_args['trainset'], numpy=True)
    trueset = load_dataset(directory + '/' + fit_args['valset'], numpy=True)
    cp_set = comformal_prediction(dkr, calset)

    def cp_set_function(x):  # dumming function. returns the cp_set given x.
        return cp_set

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

    ax.scatter(trueset['x'][:, -1, 0], trueset['x'][:, -1, 1],
               s=1, c='black', marker='o', label='Validation Set')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    # ax.legend(loc=2)
    plt.tight_layout()
    plt.show()
