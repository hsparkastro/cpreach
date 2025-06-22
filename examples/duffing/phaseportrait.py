import matplotlib.pyplot as plt
import yaml
import os
import torch
import numpy as np
from depikt import Koopman

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

np.set_printoptions(precision=3, suppress=True)

cwd = os.getcwd()
if "examples/duffing" not in cwd:
    os.chdir("./examples/duffing")

try:
    from duffing import dynamics
except ImportError:
    from examples.duffing.duffing import dynamics


# Load NN configuration and training options
config_path = "standard"
with open("config/" + config_path + ".yaml", "r") as file:
    config = yaml.safe_load(file)

model_args = config["model_args"]
fit_args = config["fit_args"]
filepath = fit_args["savepath"]
dkr = Koopman.from_args(**model_args, filepath=filepath, device=device)


def predict(x, param):

    in1 = np.logical_and(x[:, 0] >= -3.0, x[:, 0] <= 0.0)[:, None]
    in3 = np.logical_and(x[:, 0] > 0.0, x[:, 0] <= 3.0)[:, None]

    nsample = x.shape[0]
    u = np.empty((nsample, 0))
    y1 = dkr.predict(x, u, param)
    y3 = -dkr.predict(-x, u, param)

    y = y1 * in1 + y3 * in3

    return y


x_range = ([-3.0, 3.0], [-1.0, 1.0])
param_sets = (
    [0.2, -0.5, -0.2],
)

for i, param in enumerate(param_sets):

    N = 101
    X1, X2 = np.meshgrid(
        np.linspace(x_range[0][0], x_range[0][1], N),
        np.linspace(x_range[1][0], x_range[1][1], N),
    )
    x = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
    param_grid = np.array([param]).repeat(N**2, axis=0)

    y = dynamics(0, x, param_grid)

    Y1 = y[:, 0].reshape((N, N))
    Y2 = y[:, 1].reshape((N, N))
    speed = np.linalg.norm(y, axis=1).reshape((N, N))
    fig, ax = plt.subplots(figsize=(6, 3.5 * 0.75))
    strm1 = ax.streamplot(
        X1, X2, Y1, Y2, color=speed, linewidth=1.5, cmap="cool"
    )
    cbar1 = fig.colorbar(strm1.lines)
    cbar1.set_label(r"$\| \dot{x} \|$")

    # fig.tight_layout()

    y = (predict(x, param_grid) - x)/model_args['normalization']["dt"]

    Y1 = y[:, 0].reshape((N, N))
    Y2 = y[:, 1].reshape((N, N))
    # fig, ax = plt.subplots(figsize=(6,2.5))
    # strm2 = ax.streamplot(
    #     X1, X2, Y1, Y2, color=speed, linewidth=1.5, cmap="cool"
    # )
    strm2 = ax.streamplot(X1, X2, Y1, Y2, color="k", linewidth=1.3)
    strm2.lines.set_linestyles("dashed")
    # cbar2 = fig.colorbar(strm2.lines)
    # cbar2.set_label(r"$\| \dot{x} \|$")

    # vmin = min(cbar1.vmin, cbar2.vmin)
    # vmax = max(cbar1.vmax, cbar2.vmax)

    # strm1.lines.set_clim(vmin, vmax)
    # strm2.lines.set_clim(vmin, vmax)

    ax.plot([100, 101], [100, 101], color=(0.5, 0.0, 0.5), label="True")
    ax.plot(
        [100, 101],
        [100, 101],
        color="k",
        linestyle="dashed",
        label="Predicted",
    )
    ax.legend(loc="upper right", framealpha=1.0)

    ax.set_xlim([-3, 3])
    ax.set_ylim([-1, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylabel(r"$x_2$")
    ax.set_xlabel(r"$x_1$")
    plt.draw()

    ax.set_title(
        rf"$\delta$={param[0]}, $\alpha$={param[1]}, $\beta$={param[2]}"
    )
    fig.tight_layout()
    plt.savefig(
        f"figures/duffing_phaseportrait_{config_path}_{i}.png", dpi=300)
    plt.show()
    plt.close()
