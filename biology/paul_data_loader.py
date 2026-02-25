"""
Paul data loader.
"""

import torch
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
import warnings


class PaulDataLoader:
    """Data loader for Paul myeloid progenitors dataset."""

    def __init__(self, dim: int = 50):
        self.dim = dim
        self.adata = None
        self.pca = None

    def load_paul_data(self):
        """Load Paul et al. myeloid progenitors dataset."""

        adata = sc.datasets.paul15()

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata.raw = adata
        adata = adata[:, adata.var.highly_variable]
        sc.pp.scale(adata, max_value=10)

        self.adata = adata
        return adata

    def get_latent_representation(self, method: str = "pca"):
        """Get latent representation of the data."""
        if self.adata is None:
            self.load_paul_data()

        X = self.adata.X.toarray() if hasattr(self.adata.X, "toarray") else self.adata.X

        if method == "pca":
            if self.pca is None:
                self.pca = PCA(n_components=self.dim)
                X_latent = self.pca.fit_transform(X)
            else:
                X_latent = self.pca.transform(X)
        else:
            X_latent = X[:, : self.dim]

        return torch.tensor(X_latent, dtype=torch.float32)

    def get_source_target_distributions(self):
        """
        Define source and target distributions.
        """
        if self.adata is None:
            self.load_paul_data()

        X_latent = self.get_latent_representation()

        if "paul15_clusters" in self.adata.obs.columns:
            cell_types = self.adata.obs["paul15_clusters"].values
        else:
            cell_types = self.adata.obs["cell_type"].values

        source_cell_types = ["9GMP", "10GMP", "7MEP"]
        source_mask = np.isin(cell_types, source_cell_types)
        source_data = X_latent[source_mask]

        mono_cell_types = ["14Mo", "15Mo"]
        mono_mask = np.isin(cell_types, mono_cell_types)
        target_mono = X_latent[mono_mask]

        neu_cell_types = ["17Neu", "16Neu"]
        neu_mask = np.isin(cell_types, neu_cell_types)
        target_neu = X_latent[neu_mask]

        self.source_data = source_data
        self.target_mono = target_mono
        self.target_neu = target_neu

        return source_data, target_mono, target_neu
