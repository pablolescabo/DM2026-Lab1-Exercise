from langchain_core.tools import tool
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.15):
        """
        Flags near-constant terms by variance for removal.
        
        Args:
            threshold: Variance threshold below which terms are flagged for removal. Defaults to 0.15.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "DTM not built. Please run build_dtm_tool first."}
        
        if threshold < 0:
            return {"error": "Threshold must be a non-negative value."}
        
        # Compute variance efficiently for sparse matrix
        # Var(X) = E[X^2] - (E[X])^2
        mean = session.feature_matrix.mean(axis=0)
        mean_sq = session.feature_matrix.power(2).mean(axis=0)
        # Ensure variance is at least 0 to avoid floating point errors
        variances = np.maximum(0, np.array(mean_sq - np.power(mean, 2)).flatten())
        
        # Create report DataFrame
        # We keep full precision for the report to avoid losing information
        report_df = pd.DataFrame({
            "term": session.feature_names,
            "variance": variances
        })
        
        # Sort by variance descending for the report
        report_df = report_df.sort_values(by="variance", ascending=False)
        
        # Identify terms to keep
        keep_mask = variances >= threshold
        kept_terms = [term for term, keep in zip(session.feature_names, keep_mask) if keep]
        removed_terms = [term for term, keep in zip(session.feature_names, keep_mask) if not keep]
        
        summary = {
            "result_id": session.next_result_id("variance_filter"),
            "threshold": threshold,
            "kept_terms": kept_terms,
            "removed_terms": removed_terms
        }
        
        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=report_df)
        return summary

    @tool
    def pearson_filter_tool(target_class: str):
        """
        Computes Pearson correlation between each term's count and a target class.
        
        Args:
            target_class: The category name to correlate against (one-vs-rest).
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {"error": "DTM or labels not built. Please run build_dtm_tool first."}
        
        if target_class not in session.categories:
            return {"error": f"Target class '{target_class}' not found in categories."}
        
        # Create binary target vector aligned with feature_matrix rows
        target_vector = (session.labels == target_class).astype(int)
        
        # Compute Pearson correlation for each term
        correlations = []
        for i in range(session.feature_matrix.shape[1]):
            term_counts = session.feature_matrix[:, i].toarray().flatten()
            # pearsonr returns (correlation, p-value)
            corr, _ = pearsonr(term_counts, target_vector)
            correlations.append(corr)
        
        # Replace NaNs (from constant terms) with 0.0
        correlations = np.nan_to_num(correlations, nan=0.0)
        
        # Create report DataFrame
        report_df = pd.DataFrame({
            "term": session.feature_names,
            "pearson_r": correlations
        })
        
        # Sort by absolute correlation descending
        report_df = report_df.iloc[report_df["pearson_r"].abs().argsort()[::-1]]
        
        summary = {
            "result_id": session.next_result_id("pearson_filter"),
            "target_class": target_class,
            "top_correlations": report_df.head(5).to_dict(orient="records")
        }
        
        session.store_result("pearson_filter_tool", {"target_class": target_class}, summary, full_report=report_df)
        return summary

    @tool
    def spearman_filter_tool(target_class: str):
        """
        Computes Spearman correlation between each term's count and a target class.
        
        Args:
            target_class: The category name to correlate against (one-vs-rest).
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {"error": "DTM or labels not built. Please run build_dtm_tool first."}
        
        if target_class not in session.categories:
            return {"error": f"Target class '{target_class}' not found in categories."}
        
        # Create binary target vector aligned with feature_matrix rows
        target_vector = (session.labels == target_class).astype(int)
        
        # Compute Spearman correlation for each term
        correlations = []
        for i in range(session.feature_matrix.shape[1]):
            term_counts = session.feature_matrix[:, i].toarray().flatten()
            # spearmanr returns (correlation, p-value)
            corr, _ = spearmanr(term_counts, target_vector)
            correlations.append(corr)
        
        # Replace NaNs (from constant terms) with 0.0
        correlations = np.nan_to_num(correlations, nan=0.0)
        
        # Create report DataFrame
        report_df = pd.DataFrame({
            "term": session.feature_names,
            "spearman_r": correlations
        })
        
        # Sort by absolute correlation descending
        report_df = report_df.iloc[report_df["spearman_r"].abs().argsort()[::-1]]
        
        summary = {
            "result_id": session.next_result_id("spearman_filter"),
            "target_class": target_class,
            "top_correlations": report_df.head(5).to_dict(orient="records")
        }
        
        session.store_result("spearman_filter_tool", {"target_class": target_class}, summary, full_report=report_df)
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
