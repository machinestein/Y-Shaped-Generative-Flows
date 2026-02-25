# Y-Shaped Generative Flows

PyTorch implementations and experiments for **Y-Shaped Generative Flows** — continuous-time generative model that encourages **shared motion before branching** (a “Y-shape” in trajectory space).

This repo currently contains:
- a **toy notebook** for intuition and visualization
- **biology-focused experiments** (Paul myeloid progenitors + Tedsim branching differentiation), plus utilities

## Toy demo (intuition first)

Open the toy notebook:

```bash
jupyter notebook Y-Flows-Toy.ipynb
```

What you should see/learn:

* how trajectories can **merge early**
* how they **split later** (branching)
* how cost/regularization affects “how Y-shaped” the transport becomes


## Biology experiments

The biology code lives in `biology/` and is organized around datasets + reusable loaders/metrics.

* **Paul myeloid progenitors**: differentiation from progenitors to **monocytes** and **neutrophils**
* **Tedsim**: synthetic branching differentiation

### Install dependencies

Install core deps (edit as needed for your setup):

```bash
pip install torch numpy scipy moscot matplotlib scikit-learn plotly geomloss scanpy
```

#### Paul Dataset

```bash
cd biology/experiments
python paul_y_flows.py
```

#### Tedsim Dataset

```bash
cd biology/experiments
python tedsim_y_flows.py
```

### Biology utilities (as described in `biology/README.md`)

* `paul_data_loader.py` — Paul dataset preprocessing
* `tedsim_data_loader.py` — Tedsim dataset loader with caching
* `wasserstein_distances.py` — distance/metric helpers

### Outputs

Each experiment typically generates:

* trajectory visualizations
* Wasserstein distance metrics (**W1**, **W2**)
* **MMD-RBF** distances
* figures saved under `figs/`

