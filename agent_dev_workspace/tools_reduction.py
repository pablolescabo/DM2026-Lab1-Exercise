from langchain_core.tools import tool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        random_state: int = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15
    ) -> dict:
        """
        Reduces the dimensionality of the DTM to 2D using PCA, t-SNE, or UMAP.

        Args:
            method: The reduction method ("pca", "tsne", or "umap").
            random_state: Seed for reproducibility.
            perplexity: Perplexity for t-SNE.
            n_neighbors: Number of neighbors for UMAP.
        """
        if session.feature_matrix is None:
            return {"error": "Feature matrix not found. Please build the DTM first."}
        
        # Convert sparse matrix to dense for reduction
        data = session.feature_matrix.toarray()
        n_samples = data.shape[0]
        
        adjustment_note = None
        
        if method == "pca":
            reducer = PCA(n_components=2, random_state=random_state)
            coords = reducer.fit_transform(data)
            explained_variance = reducer.explained_variance_ratio_.tolist()
        elif method == "tsne":
            # Adjust perplexity if it's too large for the number of samples
            adj_perplexity = min(perplexity, max(1.0, n_samples - 1))
            if adj_perplexity != perplexity:
                adjustment_note = f"Perplexity adjusted from {perplexity} to {adj_perplexity}"
            reducer = TSNE(n_components=2, perplexity=adj_perplexity, random_state=random_state)
            coords = reducer.fit_transform(data)
            explained_variance = None
        elif method == "umap":
            # Adjust n_neighbors if it's too large for the number of samples
            adj_n_neighbors = min(n_neighbors, max(2, n_samples))
            if adj_n_neighbors != n_neighbors:
                adjustment_note = f"n_neighbors adjusted from {n_neighbors} to {adj_n_neighbors}"
            reducer = UMAP(n_components=2, n_neighbors=adj_n_neighbors, random_state=random_state)
            coords = reducer.fit_transform(data)
            explained_variance = None
        else:
            return {"error": f"Invalid method '{method}'. Supported methods are 'pca', 'tsne', and 'umap'."}

        # Build scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        # Use session.labels for coloring to ensure alignment with feature_matrix rows
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=session.labels, ax=ax)
        session.pending_figure = fig

        summary = {
            "result_id": session.next_result_id(f"dim_red_{method}"),
            "method": method,
            "coordinates": coords.tolist(),
            "explained_variance_ratio": explained_variance
        }
        
        if adjustment_note:
            summary["note"] = adjustment_note

        session.store_result("reduce_dimensions_tool", {"method": method}, summary)
        return summary

    @tool
    def binarize_labels_tool() -> dict:
        """
        One-hot encodes the category labels from session.labels and stores the result in session.artifacts.
        """
        if session.labels is None:
            return {"error": "Labels not found. Please ensure labels are set."}
        
        # Perform one-hot encoding
        binarized = pd.get_dummies(session.labels)
        
        # Store in session.artifacts
        session.artifacts["binarized_labels"] = binarized
        
        summary = {
            "result_id": session.next_result_id("binarize_labels"),
            "n_classes": binarized.shape[1],
            "classes": binarized.columns.tolist()
        }
        
        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
