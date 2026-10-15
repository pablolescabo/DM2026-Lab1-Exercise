from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def make_tools(session):
    @tool
    def build_dtm_tool(ngram_range: List[int] = [1, 1]):
        """
        Builds the document-term matrix from the working dataframe.
        
        Args:
            ngram_range: The lower and upper boundary of the range of n-values for different n-grams to be extracted.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please run load_dataset_tool first."}

        # Convert List[int] to tuple for CountVectorizer
        ngram_tuple = tuple(ngram_range)
        
        vectorizer = CountVectorizer(ngram_range=ngram_tuple)
        feature_matrix = vectorizer.fit_transform(session.dataframe["text"].fillna(""))
        
        session.feature_matrix = feature_matrix
        session.feature_names = vectorizer.get_feature_names_out().tolist()
        session.artifacts["count_vectorizer"] = vectorizer
        
        # Set labels and categories
        session.set_labels(session.dataframe["category_name"].values)
        
        # Calculate sparsity
        non_zero = feature_matrix.nnz
        total_elements = feature_matrix.shape[0] * feature_matrix.shape[1]
        sparsity_pct = 100 * (1 - non_zero / total_elements)
        
        summary = {
            "result_id": session.next_result_id("dtm"),
            "non_zero": int(non_zero),
            "total_elements": int(total_elements),
            "sparsity_pct": float(sparsity_pct)
        }
        
        session.store_result("build_dtm_tool", {"ngram_range": ngram_range}, summary)
        return summary

    @tool
    def term_frequency_tool():
        """
        Aggregates term frequencies across all documents in the document-term matrix.
        
        Returns:
            A dictionary containing the total frequency for each term.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "DTM not built. Please run build_dtm_tool first."}

        # Sum frequencies across all documents (rows)
        term_counts = session.feature_matrix.sum(axis=0).A1
        
        # Create a dictionary mapping term names to their total frequencies
        frequencies = {
            name: int(count) 
            for name, count in zip(session.feature_names, term_counts)
        }
        
        summary = {
            "result_id": session.next_result_id("term_freq"),
            "frequencies": frequencies
        }
        
        session.store_result("term_frequency_tool", {}, summary)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20):
        """
        Heatmap of a slice of the raw document-term matrix.
        
        Args:
            n_terms: Number of terms (columns) to include.
            n_documents: Number of documents (rows) to include.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "DTM not built. Please run build_dtm_tool first."}
        
        if n_terms <= 0 or n_documents <= 0:
            return {"error": "n_terms and n_documents must be positive integers."}

        # Slice the matrix
        matrix_slice = session.feature_matrix[0:n_documents, 0:n_terms].toarray()
        term_labels = session.feature_names[0:n_terms]
        doc_labels = [f"Doc {i}" for i in range(matrix_slice.shape[0])]
        
        # Create heatmap
        df_slice = pd.DataFrame(matrix_slice, index=doc_labels, columns=term_labels)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_slice, annot=True, fmt="d", ax=ax)
        session.pending_figure = fig
        
        summary = {
            "result_id": session.next_result_id("heatmap"),
            "matrix_slice": matrix_slice.tolist(),
            "terms": term_labels,
            "documents": doc_labels
        }
        
        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
