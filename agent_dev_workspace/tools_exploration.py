from langchain_core.tools import tool
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_index_a: int, doc_index_b: int) -> dict:
        """
        Calculates the cosine similarity between two documents by their indices in the DTM.

        Args:
            doc_index_a (int): The index of the first document.
            doc_index_b (int): The index of the second document.
        """
        if session.feature_matrix is None:
            return {"error": "Feature matrix not found. Please build the DTM first."}
        
        # Check for negative indices or out of bounds
        if doc_index_a < 0 or doc_index_b < 0 or \
           doc_index_a >= session.feature_matrix.shape[0] or \
           doc_index_b >= session.feature_matrix.shape[0]:
            return {"error": "Index out of bounds."}

        # Extract the rows from the feature matrix
        vec_a = session.feature_matrix[doc_index_a]
        vec_b = session.feature_matrix[doc_index_b]

        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(vec_a, vec_b)[0][0]

        summary = {
            "result_id": session.next_result_id("cosine_sim"),
            "doc_index_a": doc_index_a,
            "doc_index_b": doc_index_b,
            "similarity": float(similarity)
        }

        session.store_result("cosine_similarity_tool", {"doc_index_a": doc_index_a, "doc_index_b": doc_index_b}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool() -> dict:
        """
        Computes the Pearson correlation matrix for the top 20 terms by variance and plots a heatmap.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Feature matrix or names not found. Please build the DTM first."}

        # 1. Compute variance efficiently for sparse matrix
        mean = session.feature_matrix.mean(axis=0)
        mean_sq = session.feature_matrix.power(2).mean(axis=0)
        variances = np.maximum(0, np.array(mean_sq - np.power(mean, 2)).flatten())
        
        # 2. Get top 20 terms by variance
        top_indices = np.argsort(variances)[::-1][:20]
        top_terms = [session.feature_names[i] for i in top_indices]
        
        # 3. Extract only the top 20 columns and convert to dense
        top_matrix = session.feature_matrix[:, top_indices].toarray()

        # 4. Compute Pearson correlation matrix
        # rowvar=False means each column is a variable (term)
        corr_matrix = np.corrcoef(top_matrix, rowvar=False)
        
        # Replace NaNs (from constant terms) with 0.0
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Reset diagonal to 1.0
        np.fill_diagonal(corr_matrix, 1.0)

        # 5. Build heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, xticklabels=top_terms, yticklabels=top_terms, ax=ax)
        session.pending_figure = fig

        # 6. Prepare summary
        summary = {
            "result_id": session.next_result_id("corr_matrix"),
            "terms": top_terms,
            "correlation_matrix": corr_matrix.tolist()
        }

        session.store_result("feature_correlation_matrix_tool", {}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
