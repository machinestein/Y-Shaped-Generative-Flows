"""
Data loader for Tedsim dataset.
"""
import os
import scanpy as sc
import pandas as pd
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import moscot as mt


class TedsimDataLoader:
    """Data loader for Tedsim dataset"""

    def __init__(self, latent_dim: int = 50, cache_dir: str = "cache"):
        self.latent_dim = latent_dim
        self.adata = None
        self.scaler = StandardScaler()
        self.cache_dir = cache_dir
        self.cached_pca_data = None
        self.cached_pca_model = None
        os.makedirs(cache_dir, exist_ok=True)

    def load_tedsim_data(self):
        """Load Tedsim dataset or create synthetic version"""

        adata = mt.datasets.tedsim()
        adata.obs["time"] = pd.to_numeric(adata.obs["time"]).astype("category")

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)

        self.adata = adata
        return adata

    def _get_cache_filename(self, method: str) -> str:
        """Generate cache filename based on method and data characteristics"""
        if self.adata is not None:
            data_hash = hash(
                (self.adata.shape[0], self.adata.shape[1], method, self.latent_dim)
            )
            data_hash = abs(data_hash) % 100000
        else:
            data_hash = 0
        return os.path.join(
            self.cache_dir, f"tedsim_pca_{method}_dim{self.latent_dim}_{data_hash}.pkl"
        )

    def _save_pca_cache(
        self, pca_data: np.ndarray, method: str, additional_data: dict = None
    ):
        """Save PCA results to cache"""
        cache_file = self._get_cache_filename(method)

        cache_data = {
            "pca_data": pca_data,
            "method": method,
            "latent_dim": self.latent_dim,
            "data_shape": self.adata.shape if self.adata is not None else None,
            "additional_data": additional_data or {},
        }

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"PCA cache saved to: {cache_file}")
        except Exception as e:
            print(f"Failed to save PCA cache: {e}")

    def _load_pca_cache(self, method: str) -> tuple:
        """Load PCA results from cache with flexible file matching"""

        if not os.path.exists(self.cache_dir):
            print(f"Cache directory doesn't exist: {self.cache_dir}")
            return None, None

        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith(".pkl")]

        pattern = f"tedsim_pca_{method}_dim{self.latent_dim}_"
        matching_files = [f for f in cache_files if f.startswith(pattern)]

        if not matching_files:
            print(f"No matching cache files found for pattern: {pattern}*.pkl")
            return None, None

        matching_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(self.cache_dir, f)),
            reverse=True,
        )
        cache_file = os.path.join(self.cache_dir, matching_files[0])

        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            if (
                cache_data.get("method") == method
                and cache_data.get("latent_dim") == self.latent_dim
            ):
                print(f"PCA cache loaded from: {cache_file}")
                print(f"Cache validation successful!")
                return cache_data["pca_data"], cache_data.get("additional_data", {})
            else:
                print(f"Cache parameters don't match:")
                print(f"Method: {cache_data.get('method')} vs {method}")
                print(
                    f"Latent dim: {cache_data.get('latent_dim')} vs {self.latent_dim}"
                )
                return None, None

        except Exception as e:
            print(f"Failed to load PCA cache: {e}")
            return None, None

    def clear_pca_cache(self, method: str = None):
        """Clear PCA cache files"""
        import glob

        if method:
            cache_file = self._get_cache_filename(method)
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"Cleared PCA cache for method: {method}")
        else:
            cache_pattern = os.path.join(self.cache_dir, "tedsim_pca_*.pkl")
            cache_files = glob.glob(cache_pattern)
            for cache_file in cache_files:
                os.remove(cache_file)
            print(f"Cleared {len(cache_files)} PCA cache files")

    def get_latent_representation(self, method: str = "pca"):
        """Get latent representation of the data using scanpy preprocessing with caching"""
        if self.adata is None:
            self.load_tedsim_data()

        cached_data, additional_data = self._load_pca_cache(method)
        if cached_data is not None:
            self.cached_pca_data = cached_data
            self.last_latent_method = method
            self.last_latent_dim = cached_data.shape[1]
            return torch.tensor(cached_data, dtype=torch.float32)

        adata_copy = self.adata.copy()
        print(f"Computing {method.upper()} representation (not cached)...")

        if method == "pca":
            max_comps = min(50, adata_copy.n_vars, adata_copy.n_obs - 1)
            sc.pp.pca(adata_copy, n_comps=max_comps, random_state=0)

            X_latent = adata_copy.obsm["X_pca"][:, : self.latent_dim]

            self.dimensionality_reducer = None
            print(f"PCA completed: {X_latent.shape[1]} components")

        elif method == "umap":
            if not UMAP_AVAILABLE:
                raise ImportError(
                    "umap-learn is not installed. Install with: pip install umap-learn"
                )

            sc.pp.pca(adata_copy, n_comps=40, random_state=0)
            sc.pp.neighbors(adata_copy, n_neighbors=15, n_pcs=40)
            sc.tl.umap(adata_copy, random_state=42)

            if self.latent_dim <= 2:
                X_latent = adata_copy.obsm["X_umap"][:, : self.latent_dim]
            else:
                X_latent = adata_copy.obsm["X_pca"][:, : self.latent_dim]

            self.dimensionality_reducer = None

        else:
            max_comps = min(50, adata_copy.n_vars, adata_copy.n_obs - 1)
            sc.pp.pca(adata_copy, n_comps=max_comps, random_state=0)
            X_latent = adata_copy.obsm["X_pca"][:, : self.latent_dim]
            self.dimensionality_reducer = None

        self._save_pca_cache(X_latent, method)
        self.cached_pca_data = X_latent

        self.last_latent_method = method
        self.last_latent_dim = X_latent.shape[1]

        return torch.tensor(X_latent, dtype=torch.float32)

    def get_source_target_distributions(self):
        """
        Define source and target distributions.
        """
        if self.adata is None:
            self.load_tedsim_data()

        X_latent = self.get_latent_representation()

        state_labels = self.adata.obs["state"].astype(str).values

        source_state = "6"
        source_mask = state_labels == source_state
        source_data = X_latent[source_mask]

        target_state_1 = "1"
        target_state_2 = "2"

        target_1_mask = state_labels == target_state_1
        target_2_mask = state_labels == target_state_2

        target_1_data = X_latent[target_1_mask]
        target_2_data = X_latent[target_2_mask]

        target_data = torch.cat([target_1_data, target_2_data], dim=0)
        branch_labels = torch.cat(
            [
                torch.zeros(len(target_1_data)),
                torch.ones(len(target_2_data)),
            ],
            dim=0,
        )

        return (
            source_data,
            target_data,
            branch_labels,
            {
                "state_labels": state_labels,
                "source_mask": source_mask,
                "target_1_mask": target_1_mask,
                "target_2_mask": target_2_mask,
            },
        )
