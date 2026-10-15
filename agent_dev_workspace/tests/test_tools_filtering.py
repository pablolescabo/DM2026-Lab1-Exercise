import pytest
import pandas as pd
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools

def test_variance_filter_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # 1. Load the fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # 2. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    # 3. Run variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    result = variance_filter_tool.invoke({"threshold": 0.15})
    
    # 4. Assert against known answer #5
    # Expected variances
    expected_variances = {
        "alpha": 1.1875,
        "always": 0.109375,
        "beta": 0.234375,
        "delta": 0.25,
        "gamma": 1.109375
    }
    
    # Get the full report from the session results store
    report_df = session.results_store[result["result_id"]]["full_report"]
    
    for term, expected_var in expected_variances.items():
        actual_var = report_df.loc[report_df["term"] == term, "variance"].values[0]
        assert pytest.approx(actual_var, abs=1e-5) == expected_var
        
    # Check filtering results
    assert "always" in result["removed_terms"]
    assert "always" not in result["kept_terms"]
    assert all(term in result["kept_terms"] for term in ["alpha", "beta", "delta", "gamma"])

def test_pearson_filter_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # 1. Load the fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # 2. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    # 3. Run pearson_filter_tool
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    result = pearson_filter_tool.invoke({"target_class": "catB"})
    
    # 4. Assert against known answer #6
    expected_correlations = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307
    }
    
    report_df = session.results_store[result["result_id"]]["full_report"]
    
    for term, expected_corr in expected_correlations.items():
        actual_corr = report_df.loc[report_df["term"] == term, "pearson_r"].values[0]
        assert pytest.approx(actual_corr, abs=1e-3) == expected_corr

def test_spearman_filter_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # 1. Load the fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # 2. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    # 3. Run spearman_filter_tool
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")
    result = spearman_filter_tool.invoke({"target_class": "catB"})
    
    # 4. Assert against known answer #6
    expected_correlations = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363
    }
    
    report_df = session.results_store[result["result_id"]]["full_report"]
    
    for term, expected_corr in expected_correlations.items():
        actual_corr = report_df.loc[report_df["term"] == term, "spearman_r"].values[0]
        assert pytest.approx(actual_corr, abs=1e-3) == expected_corr
