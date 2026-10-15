from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools
import pytest

def test_build_dtm_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load the dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build the DTM
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    result = build_dtm_tool.invoke({"ngram_range": [1, 1]})
    
    # Assert against known answer #4
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5)
    assert len(session.feature_names) == 5
    assert set(session.feature_names) == {"alpha", "always", "beta", "delta", "gamma"}

def test_term_frequency_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load the dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build the DTM
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"ngram_range": [1, 1]})
    
    # Run term_frequency_tool
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")
    result = term_frequency_tool.invoke({})
    
    # Assert against known answer #14
    expected_frequencies = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }
    assert result["frequencies"] == expected_frequencies

def test_dtm_heatmap_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load the dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build the DTM
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"ngram_range": [1, 1]})
    
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")
    
    # Test with n_terms=2, n_documents=3 (Known answer #21)
    result_slice = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    expected_slice = [[3, 1], [2, 1], [1, 1]]
    assert result_slice["matrix_slice"] == expected_slice
    assert result_slice["terms"] == ["alpha", "always"]
    
    # Test with default values (should return full 8x5 matrix)
    result_full = dtm_heatmap_tool.invoke({})
    assert len(result_full["matrix_slice"]) == 8
    assert len(result_full["matrix_slice"][0]) == 5
    assert session.pending_figure is not None
