# Y-Shaped Generative Flows

This repository contains implementations of Y-shaped generative flows for different datasets: Paul myeloid progenitors, Tedsim, and LiDAR data.

## Experiments

### 1. Paul Dataset (`paul_y_flows.py`)
Y-shaped transport for Paul et al. myeloid progenitors dataset, modeling differentiation from progenitors to monocytes and neutrophils.

**Run:**
```bash
cd experiments
python paul_y_flows.py
```

### 2. Tedsim Dataset (`tedsim_y_flows.py`)
Y-shaped transport for Tedsim synthetic dataset, modeling branching differentiation.

**Run:**
```bash
cd experiments
python tedsim_y_flows.py
```

## Requirements

- Python 3.8+
- PyTorch
- scanpy
- geomloss
- matplotlib
- plotly
- scikit-learn
- moscot (for Tedsim)

## Data Loaders

- `paul_data_loader.py` - Paul dataset preprocessing
- `tedsim_data_loader.py` - Tedsim dataset with caching
- `wasserstein_distances.py` - Distance metrics utilities

## Output

Each experiment generates:
- Trajectory visualizations
- Wasserstein distance metrics (W1, W2)
- MMD-RBF distances
- Saved figures in `figs/` directory
