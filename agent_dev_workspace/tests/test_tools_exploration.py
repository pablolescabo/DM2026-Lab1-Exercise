import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools
from tools_dtm import make_tools as make_dtm_tools

def test_cosine_similarity_tool():
    # Setup session and tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load fixture
    load_dataset_tool.invoke({"file_path": "agent_dev/sample_fixture.csv", "text_column": "text", "label_column": "label"})
    
    # Build DTM (required for cosine similarity)
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    # Get cosine similarity tool
    exploration_tools = make_tools(session)
    cosine_similarity_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")
    
    # Test cases from KA #16
    # 0 vs 1: 0.9847
    res1 = cosine_similarity_tool.invoke({"doc_index_a": 0, "doc_index_b": 1})
    assert res1["similarity"] == pytest.approx(0.9847, abs=1e-4)
    
    # 0 vs 3: 0.0909
    res2 = cosine_similarity_tool.invoke({"doc_index_a": 0, "doc_index_b": 3})
    assert res2["similarity"] == pytest.approx(0.0909, abs=1e-4)
    
    # 3 vs 4: 0.9847
    res3 = cosine_similarity_tool.invoke({"doc_index_a": 3, "doc_index_b": 4})
    assert res3["similarity"] == pytest.approx(0.9847, abs=1e-4)
    
    # 0 vs 7: 0.0
    res4 = cosine_similarity_tool.invoke({"doc_index_a": 0, "doc_index_b": 7})
    assert res4["similarity"] == pytest.approx(0.0, abs=1e-4)

def test_feature_correlation_matrix_tool():
    # Setup session and tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load fixture
    load_dataset_tool.invoke({"file_path": "agent_dev/sample_fixture.csv", "text_column": "text", "label_column": "label"})
    
    # Build DTM (required for correlation matrix)
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    # Get correlation matrix tool
    exploration_tools = make_tools(session)
    feature_correlation_matrix_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")
    
    # Run tool
    res = feature_correlation_matrix_tool.invoke({})
    
    # Test cases from KA #19
    # Terms: alpha, gamma, delta, beta, always
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms
    
    # Check alpha/beta correlation (0.8885)
    # alpha is index 0, beta is index 3
    assert res["correlation_matrix"][0][3] == pytest.approx(0.8885, abs=1e-4)
    
    # Check gamma/delta correlation (0.8307)
    # gamma is index 1, delta is index 2
    assert res["correlation_matrix"][1][2] == pytest.approx(0.8307, abs=1e-4)
    
    # Check diagonal is 1.0
    assert res["correlation_matrix"][0][0] == pytest.approx(1.0, abs=1e-4)
    assert res["correlation_matrix"][1][1] == pytest.approx(1.0, abs=1e-4)
