from langchain_core.tools import tool
import pandas as pd
import numpy as np
import os
import tempfile
import locale
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from PAMI.extras.convert.DF2DB import DF2DB
from PAMI.frequentPattern.basic.FPGrowth import FPGrowth
from PAMI.frequentPattern.topk.FAE import FAE
from PAMI.frequentPattern.maximal.MaxFPGrowth import MaxFPGrowth

def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str,
        algorithm: str = "fpgrowth",
        min_sup: int = None,
        k: int = None
    ) -> dict:
        """
        Mines frequent patterns for a specific category using PAMI.

        Args:
            category_name: The category to mine.
            filtering_method: "variance", "tfidf", or "term_frequency".
            algorithm: "fpgrowth", "topk", or "maxfpgrowth".
            min_sup: Minimum support (required for fpgrowth/maxfpgrowth).
            k: Number of patterns (required for topk).
        """
        if session.dataframe is None:
            return {"error": "Dataset not loaded."}
        
        # 1. Filter and build local DTM
        df_cat = session.dataframe[session.dataframe["category_name"] == category_name]
        if df_cat.empty:
            return {"error": f"Category '{category_name}' not found or empty."}
        
        vectorizer = CountVectorizer(stop_words="english")
        dtm = vectorizer.fit_transform(df_cat["text"])
        terms = np.array(vectorizer.get_feature_names_out())
        
        # 2. Apply filtering
        if filtering_method == "variance":
            variances = np.var(dtm.toarray(), axis=0)
            p5, p95 = np.percentile(variances, [5, 95])
            mask = (variances >= p5) & (variances <= p95)
        elif filtering_method == "term_frequency":
            sums = np.array(dtm.sum(axis=0)).flatten()
            p5, p95 = np.percentile(sums, [5, 95])
            mask = (sums >= p5) & (sums <= p95)
        elif filtering_method == "tfidf":
            tfidf = TfidfTransformer().fit_transform(dtm)
            means = np.array(tfidf.mean(axis=0)).flatten()
            p20 = np.percentile(means, 20)
            mask = means >= p20
        else:
            return {"error": "Invalid filtering method."}
        
        if not np.any(mask):
            return {"patterns": [], "note": "no terms survived filtering"}
        
        filtered_dtm = dtm[:, mask]
        filtered_terms = terms[mask]
        
        # 3. Convert to transactional database
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            # Create DataFrame for DF2DB
            df_trans = pd.DataFrame(filtered_dtm.toarray(), columns=filtered_terms)
            converter = DF2DB(df_trans, 'dense')
            converter.convert2TransactionalDatabase(path, '>=', 1)
            
            # Re-encode to UTF-8
            enc = locale.getpreferredencoding()
            with open(path, 'r', encoding=enc) as f:
                content = f.read()
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 4. Mine patterns
            patterns = []
            if algorithm == "fpgrowth":
                if min_sup is None: return {"error": "min_sup required for fpgrowth"}
                obj = FPGrowth(path, minSup=min_sup)
                obj.mine()
                patterns = obj.getPatterns()
            elif algorithm == "topk":
                if k is None: return {"error": "k required for topk"}
                obj = FAE(path, k=k)
                obj.mine()
                patterns = obj.getPatterns()
            elif algorithm == "maxfpgrowth":
                if min_sup is None: return {"error": "min_sup required for maxfpgrowth"}
                obj = MaxFPGrowth(path, minSup=min_sup)
                obj.mine()
                patterns = obj.getPatterns()
            else:
                return {"error": "Invalid algorithm."}
            
            summary = {
                "result_id": session.next_result_id("patterns"),
                "patterns": patterns
            }
            session.store_result("mine_patterns_tool", {"category": category_name}, summary)
            session.artifacts[f"patterns_{category_name}_{filtering_method}_{algorithm}"] = patterns
            return summary
            
        finally:
            if os.path.exists(path):
                os.remove(path)

    return [mine_patterns_tool]
