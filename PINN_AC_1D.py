import numpy as np
import sys
import os
from pathlib import Path
import torch

from scipy import linalg

from Functions.modules import Siren
from Functions.utils import loss_func_AC


from tqdm import tqdm

import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
import time

warnings.filterwarnings("ignore")

np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


## calculation of derivatives


def features_calc(X, u):

    u_t = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 1:2]
    u_x = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0][:, 0:1]
    u_xx = torch.autograd.grad(
        u_x, X, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]

    features = torch.cat([u_t, u_xx], dim=1)
    return features


## equation model
class params_equation(torch.nn.Module):
    def __init__(self, num_params):
        super().__init__()

        self.params = torch.zeros((num_params, 1))
        self.params = torch.nn.Parameter(self.params)

    def forward(self, features, u_pred):
        residual = (
            features[:, 1:2] * self.params[0]
            + features[:, 0:1]
            + self.params[1] * (torch.pow((u_pred), 3) - u_pred)
        )

        return residual


## Training parameters


num_params = 2  ## number of equation parameters to be learned
learning_rate_inr = (
    1e-5  ## learning rate value for Siren network of primary variable phi
)
learning_rate_mu = (
    1e-5  ## learning rate value for Siren network of auxilliary vaiable mu
)
learning_rate_params = 1e-4  ## learning rate value for the equation parameters


perc = 1.0  ## percentage of the data considered
noise = 0.3  ## percentage of noise considered

in_features = 2  ## number of input features to the neural network (for 1D spatio temporal case, use 2: for 2D spatio temporal case, use 3)
out_features = 1  ## number of output features
hidden_features_phi = 128  ## hidden features for the primary SIREN network
hidden_features_mu = 128  ## hidden features for the auxilliary SIREN network
hidden_layers = 3  ## number of hidden layers for both networks
batch_size = 4096  ## batch size for the training
num_epochs = 5000  ## total number of epoch for training

### file saving parameters

string_f = (
    "_lr_"
    + str(learning_rate_inr)
    + "_hf_phi_"
    + str(hidden_features_phi)
    + "_layers_"
    + str(int(hidden_layers))
    + "_ep_"
    + str(int(num_epochs))
    + "_noise_"
    + str(int(100 * noise))
    + "_perc_"
    + str(int(100 * perc))
)

### change the string_f naming accordingly when there is a change in learning scheme selected to differentiate among the results from different learning schemes.
result_path = "result_AC_1d/results" + string_f
p = Path(result_path)
if not p.exists():
    os.mkdir(result_path)
## File saving parameters
filename_u = (
    result_path + "/" + "U_data" + string_f + ".npy"
)  ## path where the primary variable phi data is saved
filename_mu = (
    result_path + "/" + "mu_pred" + string_f + ".npy"
)  ## path where the auxilliary variable mu data is saved
filename_l = (
    result_path + "/" + "Loss_collect" + string_f + ".npy"
)  ## path where the loss data for every epoch is saved
filename_p = (
    result_path + "/" + "params_collect" + string_f + ".npy"
)  ## path where the parameter data for every epoch is saved
filename_model_u = (
    result_path + "/" + "model_u" + string_f + ".pt"
)  ## path where the primary SIREN network data is saved
filename_model_mu = (
    result_path + "/" + "model_mu" + string_f + ".pt"
)  ## path where the auxilliary SIREN netowrk data is saved

pltname_u = (
    result_path + "/" + "SOC" + string_f
)  ## path where the plots of primary variable phi is saved
pltname_l = (
    result_path + "/" + "LP" + string_f + ".png"
)  ## path where loss plot is saved

pltname_p1 = (
    result_path + "/" + "PP_1" + string_f + ".png"
)  ## pltname_p1 and pltname_p2 are paths where the plots of learned parameters are saved
pltname_p2 = result_path + "/" + "PP_2" + string_f + ".png"


# loading the data
data = scipy.io.loadmat("data/AC.mat")


## preparing and normalizing the input and output data
t = data["tt"].flatten()[0:201, None]

x = data["x"].flatten()[:, None]

### use the below if the data is not normalized already
# min_t = t.min()
# max_t = t.max()

# t_norm = ((t-min_t)/(max_t-min_t)-0.5)*2


# min_x = x.min()
# max_x = x.max()

# x_norm = ((x-min_x)/(max_x-min_x)-0.5)*2


Exact = np.real(data["uu"])

X, T = np.meshgrid(x, t, indexing="ij")


X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

u_star = Exact.flatten()[:, None]

N_u = u_star.shape[0]


### adding noise


print("adding noise")

u_star_noisy = u_star + noise * np.std(u_star) * np.random.randn(
    u_star.shape[0], u_star.shape[1]
)
print("create training set")
print(np.std(u_star))
N_u = round(perc * (X_star.shape[0]))
print(N_u)
idx = np.random.choice(X_star.shape[0], N_u, replace=False)
X_train = X_star[idx, :]
u_train = u_star_noisy[idx, :]
U_noisy = griddata(X_star, u_star_noisy, (X, T), method="nearest")


# siren model initialization
model_inr = Siren(
    in_features=in_features,
    out_features=out_features,
    hidden_features=hidden_features_phi,
    hidden_layers=hidden_layers,
    outermost_linear=True,
).to(device)


print(model_inr)

# equation model initialization
model_params = params_equation(num_params=num_params).to(device)
print(model_params)


# optimizer
optim_adam = torch.optim.Adam(
    [
        {
            "params": model_inr.parameters(),
            "lr": learning_rate_inr,
            "weight_decay": 1e-6,
        },
        {
            "params": model_params.parameters(),
            "lr": learning_rate_params,
            "weight_decay": 1e-6,
        },
    ]
)


### learnig scheduler cyclic damping
# scheduler = torch.optim.lr_scheduler.CyclicLR(optim_adam, base_lr=0.1*learning_rate_inr, max_lr=10*learning_rate_inr, cycle_momentum=False, mode='exp_range', step_size_up=1000)

### learnig scheduler cyclic
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optim_adam,
    base_lr=0.1 * learning_rate_inr,
    max_lr=10 * learning_rate_inr,
    cycle_momentum=False,
)

### change the name of the file in string_f according to the learning scheme to differentiate the results for different learning schemes.
### For a constant learning rate, please comment out line (scheduler.step())

# converting numpy to torch
X_t = torch.tensor(X_train, requires_grad=True).float().to(device)
Y_t = torch.tensor(u_train).float().to(device)


# dataloader
print("now starting loading data")
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_t, Y_t), batch_size=batch_size, shuffle=True
)


Loss_collect = np.zeros((num_epochs, 3))
params_collect = np.zeros((num_epochs, num_params))


print("Parameters are learned")
print("Training Begins !!")

for epoch in range(num_epochs):
    # print(f'epoch {epoch}')
    loss_epoch = 0
    loss_data_epoch = 0
    loss_eq_epoch = 0

    ii = 0

    start_time = time.time()

    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for batch_idx, (local_batch, output) in loop:

        u_pred = model_inr(local_batch)

        ## features calculation
        features = features_calc(local_batch, u_pred)

        ## equation residual
        residual = model_params(features, u_pred)

        ## loss evaluation
        loss, loss_data, loss_eq = loss_func_AC(output, u_pred, residual)

        # Backward and optimize
        optim_adam.zero_grad()
        loss.backward()
        optim_adam.step()

        # scheduler step
        scheduler.step()
    if epoch % 1 == 0:
        print(
            "It: %d, Loss: %.3e, Loss_data: %.3e, Loss_eq: %.3e, Lambda_1: %.3e, Lambda_2: %.3e"
            % (
                epoch,
                loss.item(),
                loss_data.item(),
                loss_eq.item(),
                model_params.params[0].item(),
                model_params.params[1].item(),
            )
        )
        Loss_collect[epoch, 0] = loss.item()
        Loss_collect[epoch, 1] = loss_data.item()
        Loss_collect[epoch, 2] = loss_eq.item()
        params_collect[epoch, 0] = model_params.params[0].item()
        params_collect[epoch, 1] = model_params.params[1].item()


X_star = torch.tensor(X_star, requires_grad=True).float().to(device)


u_pred_total = model_inr(X_star)


u_pred_total = u_pred_total.cpu()
u_pred_total = u_pred_total.detach().numpy()

X_star = X_star.cpu()
X_star = X_star.detach().numpy()

U_pred = griddata(X_star, u_pred_total, (X, T), method="nearest")

U_pred = U_pred[:, :, 0]

error = np.abs(U_pred - Exact) / linalg.norm(Exact, "fro")

u_data = {"orig": Exact, "pinn": U_pred, "noisy": U_noisy, "error": error}
np.save(filename_u, u_data)

np.save(filename_l, Loss_collect)
np.save(filename_p, params_collect)

torch.save(model_inr, filename_model_u)


fig2 = plt.figure(figsize=(6, 6))
shape_loss = Loss_collect.shape[1]
if shape_loss == 1:
    plt.semilogy(Loss_collect[:, 0:1])
    plt.xlabel("Epoch")
    plt.ylabel("L2 Loss")
    plt.savefig(pltname_l)
if shape_loss == 3:
    plt.semilogy(Loss_collect[:, 0:1], label="Total loss")
    plt.semilogy(Loss_collect[:, 1:2], label="Data loss")
    plt.semilogy(Loss_collect[:, 2:3], label="Equation loss")
    plt.legend(loc="upper right", prop={"size": 17}, frameon=False)
    plt.xlabel("Epoch")
    plt.ylabel("L2 Loss")
    plt.savefig(pltname_l)
if shape_loss == 4:
    plt.semilogy(Loss_collect[:, 0:1], label="Total loss")
    plt.semilogy(Loss_collect[:, 1:2], label="Data loss")
    plt.semilogy(Loss_collect[:, 2:3], label="Equation loss")
    plt.semilogy(Loss_collect[:, 3:4], label="$\mu$ loss")
    plt.legend(loc="upper right", prop={"size": 17}, frameon=False)
    plt.xlabel("Epoch")
    plt.ylabel("L2 Loss")
    plt.savefig(pltname_l)

fig = plt.figure(figsize=(20, 5))
ax = fig.add_subplot(141)


h0 = ax.imshow(
    U_noisy,
    interpolation="nearest",
    extent=[t.min(), t.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h0, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_title("Noisy")
ax.set_xlabel("$t$", size=20)
ax.set_ylabel("$x$", size=20)

ax = fig.add_subplot(142)
h1 = ax.imshow(
    Exact,
    interpolation="nearest",
    extent=[t.min(), t.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h1, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_title("Original")
ax.set_xlabel("$t$", size=20)
ax.set_ylabel("$x$", size=20)

ax = fig.add_subplot(143)
h2 = ax.imshow(
    U_pred,
    interpolation="nearest",
    extent=[t.min(), t.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h2, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_title("SIREN representation")
ax.set_xlabel("$t$", size=20)
ax.set_ylabel("$x$", size=20)

ax = fig.add_subplot(144)
h3 = ax.imshow(
    error,
    interpolation="nearest",
    extent=[t.min(), t.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(h3, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_title("error")
ax.set_xlabel("$t$", size=20)
ax.set_ylabel("$x$", size=20)

fig.tight_layout(pad=3.0)
plt.savefig(pltname_u + ".png")


fig5 = plt.figure(figsize=(6, 6))
plt.plot(params_collect[:, 0], label=("$\lambda_1$"))
plt.yscale("symlog", linthreshy=1e-6)

plt.legend(loc="upper right", prop={"size": 17}, frameon=False)
plt.xlabel("Epoch")
plt.ylabel("Parameter")
plt.savefig(pltname_p1)


fig6 = plt.figure(figsize=(6, 6))
plt.semilogy(params_collect[:, 1], label=("$\lambda_2$"))
plt.legend(loc="lower right", prop={"size": 17}, frameon=False)
plt.xlabel("Epoch")
plt.ylabel("Parameter")
plt.savefig(pltname_p2)
