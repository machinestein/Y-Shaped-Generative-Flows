"""
Y-Flows for Tedsim dataset.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_WARNINGS"] = "off"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scanpy as sc
import pandas as pd
from geomloss import SamplesLoss
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
import warnings
import time
import gc
import os
from sklearn.decomposition import PCA

from typing import Tuple, List, Optional, Dict, Any

from wasserstein_distances import wasserstein_distance, mmd_distance
from tedsim_data_loader import TedsimDataLoader

warnings.filterwarnings("ignore")


def sample_prior(batch):
    """Sample from source distribution (early progenitors)"""
    n_source = source_data.shape[0]
    indices = torch.randint(0, n_source, (batch,), device=device)
    return source_data[indices]


def sample_data(batch):
    """Sample from target distributions (both branches)"""
    all_targets = torch.cat([target_type_a, target_type_b], dim=0)
    n_targets = all_targets.shape[0]
    indices = torch.randint(0, n_targets, (batch,), device=device)
    return all_targets[indices]


class TimeEmbedding(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
        )

    def forward(self, t):
        return self.lin(t)


class VelocityNet(nn.Module):
    def __init__(self, x_dim, time_emb=32, hidden=256):
        super().__init__()
        self.time_emb = TimeEmbedding(1, time_emb)
        self.net = nn.Sequential(
            nn.Linear(x_dim + time_emb, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x, t):
        te = self.time_emb(t)
        inp = torch.cat([x, te], dim=-1)
        return self.net(inp)


def jacobian_frobenius_squared(v_fn, x, t):
    x = x.requires_grad_(True)
    t = t.requires_grad_(False)
    v = v_fn(x, t)
    B, D = v.shape
    norms = torch.zeros(B, device=x.device)

    for i in range(D):
        grad_i = torch.autograd.grad(v[:, i].sum(), x, create_graph=True)[0]  # (B, D)
        norms += (grad_i**2).sum(dim=1)
    return norms


def integrate_ode(x0, model, steps, alpha):
    x = x0
    batch = x.shape[0]
    total_potential = 0.0
    DT = 1.0 / steps
    for k in range(steps):
        t = torch.full((batch, 1), (k / steps), device=x.device, dtype=x.dtype)
        v = model(x, t)
        pot = v.norm(dim=1) ** alpha
        total_potential = total_potential + pot.mean()
        x = x + DT * v
    return x, total_potential * DT


def visualize_y_flows_trajectories(model, n_particles=500, save_path=None):
    """
    Visualize transport.
    """

    model.eval()

    torch.manual_seed(42)
    n_source = min(n_particles, source_data.shape[0])
    indices = torch.randperm(source_data.shape[0])[:n_source]
    source_sample = source_data[indices].to(device)

    trajectory = []
    x = source_sample
    trajectory.append(x.detach().cpu().numpy())

    for k in range(steps):
        t = torch.full((n_source, 1), (k / steps), device=x.device, dtype=x.dtype)
        v = model(x, t)
        x = x + dt * v
        trajectory.append(x.detach().cpu().numpy())

    x_final = x

    target_data = torch.cat([target_type_a, target_type_b], dim=0)
    reference_points = torch.cat([source_data, target_data], dim=0)
    pca_2d = PCA(n_components=2, random_state=42)
    pca_2d.fit(reference_points.cpu().numpy())

    source_2d = pca_2d.transform(source_data.cpu().numpy())
    target_2d = pca_2d.transform(target_data.cpu().numpy())

    trajectory_2d = []
    for traj_step in trajectory:
        traj_2d = pca_2d.transform(traj_step)
        trajectory_2d.append(traj_2d)
    final_2d = trajectory_2d[-1]

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")

    n_type_a = target_type_a.shape[0]
    type_a_mask = torch.zeros(target_data.shape[0], dtype=torch.bool)
    type_a_mask[:n_type_a] = True
    type_b_mask = ~type_a_mask

    ax.scatter(
        source_2d[:, 0],
        source_2d[:, 1],
        c="#87CEEB",
        s=20,
        alpha=1.0,
        label="Source",
        zorder=3,
        edgecolor="#2E4B8F",
        linewidth=0.5,
    )

    if type_a_mask.sum() > 0:
        target_type_a_2d = target_2d[type_a_mask.cpu()]
        ax.scatter(
            target_type_a_2d[:, 0],
            target_type_a_2d[:, 1],
            c="#191970",
            s=20,
            alpha=1.0,
            label="Target (Branch 1)",
            zorder=3,
            edgecolor="#0F172A",
            linewidth=0.5,
        )

    if type_b_mask.sum() > 0:
        target_type_b_2d = target_2d[type_b_mask.cpu()]
        ax.scatter(
            target_type_b_2d[:, 0],
            target_type_b_2d[:, 1],
            c="#8A2BE2",
            s=20,
            alpha=1.0,
            label="Target (Branch 2)",
            zorder=3,
            edgecolor="#4C1D95",
            linewidth=0.5,
        )

    n_traj_show = min(200, n_source)

    if type_a_mask.sum() > 0:
        type_a_center_2d = target_2d[type_a_mask.cpu()].mean(axis=0)
    else:
        type_a_center_2d = np.array([0, 0])

    if type_b_mask.sum() > 0:
        type_b_center_2d = target_2d[type_b_mask.cpu()].mean(axis=0)
    else:
        type_b_center_2d = np.array([0, 0])

    for i in range(0, n_traj_show, 2):
        traj_x = [traj[i, 0] for traj in trajectory_2d]
        traj_y = [traj[i, 1] for traj in trajectory_2d]

        final_pos = np.array([traj_x[-1], traj_y[-1]])
        dist_to_type_a = np.linalg.norm(final_pos - type_a_center_2d)
        dist_to_type_b = np.linalg.norm(final_pos - type_b_center_2d)

        if dist_to_type_a < dist_to_type_b:
            traj_color = "#4682B4"
        else:
            traj_color = "#9370DB"

        ax.plot(
            traj_x,
            traj_y,
            color=traj_color,
            alpha=0.6,
            linewidth=1.2,
            zorder=1,
        )

    ax.scatter(
        final_2d[:, 0],
        final_2d[:, 1],
        c="#4169E1",
        s=25,
        alpha=0.8,
        label="Transported Particles",
        marker="D",
        zorder=4,
        edgecolor="white",
        linewidth=0.3,
    )

    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel("PC 2", fontsize=12)

    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=False, fontsize=11)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_facecolor("white")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        os.makedirs("figs", exist_ok=True)
        default_path = f"figs/tedsim_y_flows_alpha={alpha}.png"
        plt.savefig(default_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()

    model.train()
    return None


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dim = 50
    alpha = 1
    epsilon = 0.1

    gamma1 = 0.6
    gamma2 = 1.4

    lr = 1e-3
    steps = 15
    dt = 1.0 / steps
    batch = 256
    n_iters = 300

    lambda_sinkhorn = 1.0
    lambda_energy = 1.0
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_loader = TedsimDataLoader(latent_dim=dim)
    (
        source_data,
        target_data,
        branch_labels,
        metadata,
    ) = data_loader.get_source_target_distributions()

    n_type_a = metadata["target_1_mask"].sum()
    target_type_a = target_data[:n_type_a]
    target_type_b = target_data[n_type_a:]

    model = VelocityNet(dim, time_emb=32, hidden=256).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, backend="tensorized")

    for it in range(n_iters):
        iter_start = time.time()
        optim.zero_grad()

        sample_start = time.time()
        z = sample_prior(batch)
        sample_time = time.time() - sample_start

        xT, pot_term = integrate_ode(z, model, steps=steps, alpha=alpha)

        x_data = sample_data(batch)
        loss_sink = sinkhorn_loss(xT, x_data)

        loss = lambda_sinkhorn * loss_sink + lambda_energy * pot_term

        loss.backward()
        optim.step()

        if it % 10 == 0 or it == 1:
            print(
                f"Iter {it} | Loss {loss.item():.6f} | Sinkhorn {loss_sink.item():.6f} | Pot {pot_term.item():.6f}"
            )

    model.eval()

    z_final = sample_prior(2000)
    x_final, _ = integrate_ode(z_final, model, steps=steps, alpha=alpha)

    x_real_final = sample_data(2000)

    final_wasserstein = sinkhorn_loss(x_final, x_real_final)

    w1_overall = wasserstein_distance(
        x_final.cpu(), x_real_final.cpu(), power=1, method="exact"
    )
    w2_overall = wasserstein_distance(
        x_final.cpu(), x_real_final.cpu(), power=2, method="exact"
    )

    mmd_overall = mmd_distance(x_final.cpu(), x_real_final.cpu(), sigma=1.0)

    print(f"W1: {w1_overall:.6f}, W2: {w2_overall:.6f}, MMD-RBF: {mmd_overall:.6f}")
    visualize_y_flows_trajectories(model, n_particles=1000)
