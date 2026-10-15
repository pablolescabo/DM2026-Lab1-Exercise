from langchain_core.tools import tool
import nltk
import os

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def make_tools(session):
    @tool
    def tokenize_tool():
        """
        Tokenizes each document's text into unigrams using nltk.word_tokenize.
        
        Returns:
            A dictionary containing the number of documents tokenized and the total number of tokens.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please run load_dataset_tool first."}
        
        def tokenize_text(text):
            if not text or str(text).strip() == "":
                return []
            return nltk.word_tokenize(str(text).lower())
        
        session.dataframe["unigrams"] = session.dataframe["text"].apply(tokenize_text)
        
        total_tokens = int(session.dataframe["unigrams"].apply(len).sum())
        
        summary = {
            "result_id": session.next_result_id("tokenize"),
            "n_documents": int(len(session.dataframe)),
            "total_tokens": total_tokens
        }
        
        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "."):
        """
        Lists files and directories in the specified subdirectory of the repository.
        
        Args:
            subdirectory: The directory to list contents of. Defaults to the current directory.
        """
        # Normalize path and ensure it stays within the current working directory
        base_path = os.path.abspath(".")
        target_path = os.path.abspath(os.path.join(".", subdirectory))
        
        if not target_path.startswith(base_path):
            return {"error": "Access denied: Path is outside the workspace."}
            
        if not os.path.exists(target_path):
            return {"error": f"Path '{subdirectory}' does not exist."}
            
        if not os.path.isdir(target_path):
            return {"error": f"Path '{subdirectory}' is not a directory."}
        
        items = os.listdir(target_path)
        directories = sorted([i for i in items if os.path.isdir(os.path.join(target_path, i))])
        files = sorted([i for i in items if os.path.isfile(os.path.join(target_path, i))])
        
        summary = {
            "result_id": session.next_result_id("list_files"),
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }
        
        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def sample_data_tool(n: int, random_state: int = 42):
        """
        Subsamples the working dataframe to n rows.
        WARNING: This is a destructive operation that updates the working dataframe.
        
        Args:
            n: The number of rows to sample. Must be greater than 0.
            random_state: Seed for reproducibility.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please run load_dataset_tool first."}
        
        if n <= 0:
            return {"error": "Number of samples (n) must be greater than 0."}
            
        if n > len(session.dataframe):
            return {"error": f"Requested {n} samples, but only {len(session.dataframe)} rows available."}
            
        sampled_df = session.dataframe.sample(n=n, random_state=random_state)
        picked_indices = sampled_df.index.tolist()
        session.dataframe = sampled_df
        
        summary = {
            "result_id": session.next_result_id("sample"),
            "n_samples": n,
            "random_state": random_state,
            "picked_indices": picked_indices,
            "remaining_rows": len(session.dataframe)
        }
        
        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool():
        """
        Computes descriptive statistics for document text length and generates a box plot.
        
        Returns:
            A dictionary containing the descriptive statistics for text length overall and per category.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please run load_dataset_tool first."}
        
        # 1. Compute text length
        session.dataframe["text_length"] = session.dataframe["text"].apply(len)
        
        # 2. Compute stats
        overall_stats = session.dataframe["text_length"].describe().to_dict()
        
        # Group by category and convert to list of records
        category_stats = session.dataframe.groupby("category_name")["text_length"].describe().reset_index().to_dict(orient="records")
        
        # 3. Build box plot
        fig, ax = plt.subplots()
        sns.boxplot(data=session.dataframe, x="category_name", y="text_length", ax=ax)
        ax.set_title("Text Length by Category")
        session.pending_figure = fig
        
        summary = {
            "result_id": session.next_result_id("describe"),
            "overall_stats": overall_stats,
            "category_stats": category_stats
        }
        
        session.store_result("describe_data_tool", {}, summary)
        return summary

    @tool
    def check_missing_tool():
        """
        Checks for missing, empty, or whitespace-only text values in the working dataframe.
        
        Returns:
            A dictionary containing the count of rows with missing or empty text.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please run load_dataset_tool first."}
        
        # Count rows where text is NaN, empty, or only whitespace
        missing_count = int(
            session.dataframe["text"].isna().sum() + 
            (session.dataframe["text"].astype(str).str.strip() == "").sum()
        )
        
        summary = {
            "result_id": session.next_result_id("missing"),
            "missing_count": missing_count
        }
        
        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False):
        """
        Checks for duplicate documents in the working dataframe.
        
        Args:
            drop: If True, drops all copies of duplicated rows (keep=False).
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please run load_dataset_tool first."}
        
        # Identify duplicates based on the 'text' column
        # Use keep='first' to count the number of extra copies
        duplicates = session.dataframe.duplicated(subset=["text"], keep='first')
        duplicate_count = int(duplicates.sum())
        
        remaining_rows = int(len(session.dataframe))
        
        if drop:
            # Use keep=False to drop all copies of duplicated rows as per contract
            session.dataframe.drop_duplicates(subset=["text"], keep=False, inplace=True)
            remaining_rows = int(len(session.dataframe))
            
        summary = {
            "result_id": session.next_result_id("duplicates"),
            "duplicate_count": duplicate_count,
            "dropped": drop,
            "remaining_rows": remaining_rows
        }
        
        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5, max_chars: int = 50):
        """
        Peeks at the first n rows of the working DataFrame, truncating long text.
        
        Args:
            n: Number of rows to display. Defaults to 5.
            max_chars: Maximum characters to show for text columns. Defaults to 50.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please run load_dataset_tool first."}
        
        if n <= 0:
            return {"error": "Number of rows (n) must be greater than 0."}
        
        if max_chars <= 0:
            return {"error": "Maximum characters (max_chars) must be greater than 0."}
        
        # Create a copy for inspection so we don't modify the actual session.dataframe
        df_inspect = session.dataframe.copy()
        
        # Truncate text columns if they exist
        if "text" in df_inspect.columns:
            df_inspect["text"] = df_inspect["text"].apply(
                lambda x: (str(x)[:max_chars] + '...') if len(str(x)) > max_chars else str(x)
            )
            
        summary = {
            "result_id": session.next_result_id("inspect"),
            "shape": list(session.dataframe.shape),
            "columns": session.dataframe.columns.tolist(),
            "head": df_inspect.head(n).to_dict(orient="records")
        }
        
        session.store_result("inspect_data_tool", {"n": n, "max_chars": max_chars}, summary)
        return summary

    return [tokenize_tool, list_files_tool, sample_data_tool, describe_data_tool, check_missing_tool, check_duplicates_tool, inspect_data_tool]
